"""Durable job state.

Job progress used to live only in a module-level dict, so a server restart lost
an entire in-flight run and made finished exports unreachable. Every job now
owns a directory on disk that is updated after each applicant, which means:

* a crash keeps everything already scored,
* the export is always current instead of being written only at the very end,
* an interrupted run can resume instead of restarting from zero.

Writes are atomic (temp file + os.replace) so a crash mid-write cannot leave a
truncated state file behind.
"""

import csv
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JOBS_DIR = Path("jobs")

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def state_path(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


def results_csv_path(job_id: str) -> Path:
    return job_dir(job_id) / "results.csv"


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def create(job_id: str, csv_files: List[str], resumes_dir: Optional[str],
           total: int = 0) -> Dict[str, Any]:
    state = {
        "job_id": job_id,
        "status": STATUS_QUEUED,
        "total_files": total,
        "processed_files": 0,
        "succeeded": 0,
        "failed": 0,
        "results": [],
        "file_paths": csv_files,
        "resumes_dir": resumes_dir,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    save(state)
    return state


def save(state: Dict[str, Any]) -> None:
    """Persist job state atomically."""
    state["updated_at"] = _now()
    with _lock:
        try:
            _atomic_write(
                state_path(state["job_id"]),
                json.dumps(state, indent=2, ensure_ascii=False, default=str),
            )
        except Exception as e:
            # Persistence failing must not abort a run that is otherwise fine.
            logger.error("Could not persist state for job %s: %s", state["job_id"], e)


def load(job_id: str) -> Optional[Dict[str, Any]]:
    path = state_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Could not read state for job %s: %s", job_id, e)
        return None


def list_jobs() -> List[Dict[str, Any]]:
    """All known jobs, newest first."""
    if not JOBS_DIR.exists():
        return []
    out = []
    for d in JOBS_DIR.iterdir():
        if d.is_dir():
            state = load(d.name)
            if state:
                out.append(state)
    return sorted(out, key=lambda s: s.get("created_at") or "", reverse=True)


def append_result_row(job_id: str, row: Dict[str, Any]) -> None:
    """Append one scored applicant to the job's CSV as soon as it is ready.

    Writing incrementally means the download is valid at every moment, rather
    than only existing once the whole batch finishes.
    """
    path = results_csv_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()),
                                    extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        logger.error("Could not append CSV row for job %s: %s", job_id, e)


def rewrite_results_csv(job_id: str, rows: List[Dict[str, Any]]) -> None:
    """Rewrite the whole CSV, e.g. after a resume run reorders results."""
    if not rows:
        return
    path = results_csv_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    buf = []
    try:
        import io

        sio = io.StringIO()
        writer = csv.DictWriter(sio, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        buf = sio.getvalue()
        _atomic_write(path, buf)
    except Exception as e:
        logger.error("Could not rewrite CSV for job %s: %s", job_id, e)


def mark_stale_jobs_interrupted() -> int:
    """On startup, flag jobs that were mid-flight when the process died.

    Without this a killed run stays 'processing' forever and the UI polls it
    indefinitely -- which is exactly what made a dead server look like a frozen
    progress bar.
    """
    count = 0
    for state in list_jobs():
        if state.get("status") in (STATUS_PROCESSING, STATUS_QUEUED):
            state["status"] = STATUS_INTERRUPTED
            state["error"] = (
                "Processing stopped because the server exited. "
                "Results scored before the interruption were preserved."
            )
            save(state)
            count += 1
    return count


def scored_names(state: Dict[str, Any]) -> set:
    """Applicants already scored successfully -- skipped when resuming."""
    return {
        r.get("candidate_name")
        for r in state.get("results", [])
        if r.get("success") and r.get("candidate_name")
    }
