"""FastAPI front end for the application screener.

This module is intentionally thin: it handles HTTP, file intake and job
orchestration, and delegates the real work to focused modules --
`intake` (parse the form, match resumes), `extraction` (read a resume of any
format), `evaluator` (score it), `scoring` (one definition of the total) and
`jobstore` (durable state).
"""

import json
import logging
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

import jobstore
import preflight
from config import provider_for
from evaluator import ResumeEvaluator
from extraction import extract_resume_text
from intake import (
    GPA_THRESHOLD,
    RESUME_MATCHED,
    RESUME_UNREADABLE,
    Applicant,
    IntakeError,
    build_scoring_text,
    load_applicants,
)
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from scoring import compute_total
from transform import evaluation_to_csv_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _startup()
    yield


app = FastAPI(
    title="Application Screener", lifespan=lifespan
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
jobstore.JOBS_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED_SUFFIXES = {".csv", ".zip"}


class RubricConfig(BaseModel):
    motivation_fit: float = 25
    collaboration_perspective: float = 25
    values_judgment: float = 25
    commitments_experience: float = 25
    gpa_threshold: float = GPA_THRESHOLD


def _startup():
    """Flag jobs that were mid-flight when the process last died.

    Without this, a killed run stays 'processing' forever and the UI polls it
    indefinitely -- which is what made a dead server look like a frozen bar.
    """
    stale = jobstore.mark_stale_jobs_interrupted()
    if stale:
        logger.warning("Marked %s interrupted job(s) from a previous run", stale)

    try:
        cfg = provider_for(DEFAULT_MODEL)
        ctx = preflight.probe_context(cfg["base_url"], DEFAULT_MODEL)
        if ctx.known:
            logger.info(
                "Model %s serving %s-token context (%s)",
                DEFAULT_MODEL, ctx.context_length, ctx.source,
            )
            if ctx.context_length < 8192:
                logger.warning(
                    "Context window is only %s tokens. %s",
                    ctx.context_length, preflight.remediation(8192),
                )
        else:
            logger.info("Context window could not be probed (%s)", ctx.source)
    except Exception as e:
        logger.warning("Preflight context probe failed: %s", e)


def _resume_text_for(applicant: Applicant, provider, model: str) -> None:
    """Attach resume text, recording *why* it is missing when it is."""
    if applicant.resume_status != RESUME_MATCHED or not applicant.resume_path:
        return
    result = extract_resume_text(
        applicant.resume_path, provider=provider, vision_model=model
    )
    if result.ok:
        applicant.resume_text = result.text
    else:
        applicant.resume_status = RESUME_UNREADABLE
        applicant.resume_note = result.error or "Resume could not be read"
        logger.warning(
            "Resume unreadable for %s (%s): %s",
            applicant.name, applicant.resume_path.name, applicant.resume_note,
        )


def _result_record(applicant: Applicant, evaluation=None, error=None) -> dict:
    """Uniform result shape.

    GPA and resume metadata are attached whether or not scoring succeeded, so a
    failed row still carries everything a human reviewer needs.
    """
    return {
        "candidate_name": applicant.name,
        "email": applicant.email,
        "year": applicant.year,
        "gpa_raw": applicant.gpa_raw,
        "gpa_value": applicant.gpa_value,
        "gpa_status": applicant.gpa_status,
        "resume_status": applicant.resume_status,
        "resume_note": applicant.resume_note,
        "resume_file": applicant.resume_path.name if applicant.resume_path else "",
        "warnings": applicant.warnings,
        "evaluation": evaluation.model_dump() if evaluation else None,
        "total_score": compute_total(evaluation) if evaluation else 0.0,
        "success": evaluation is not None,
        "error": error,
    }


def process_job(job_id: str, csv_paths: List[str], resumes_dir: Optional[str] = None):
    """Score every applicant, persisting after each one.

    Setup that can fail (reading the CSV, constructing the evaluator) happens
    before the loop and marks the job failed with a clear reason. Per-applicant
    failures are isolated so one bad row never costs the rest of the batch.
    """
    state = jobstore.load(job_id) or jobstore.create(job_id, csv_paths, resumes_dir)
    state["status"] = jobstore.STATUS_PROCESSING
    state["error"] = None
    jobstore.save(state)

    try:
        applicants: List[Applicant] = []
        for csv_path in csv_paths:
            applicants.extend(load_applicants(csv_path, resumes_dir))

        model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
        evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)
        provider = evaluator.provider
    except IntakeError as e:
        state.update(status=jobstore.STATUS_FAILED, error=str(e))
        jobstore.save(state)
        logger.error("Job %s could not start: %s", job_id, e)
        return
    except Exception as e:
        state.update(
            status=jobstore.STATUS_FAILED,
            error=f"Could not initialize scoring: {e}",
        )
        jobstore.save(state)
        logger.exception("Job %s failed during setup", job_id)
        return

    already = jobstore.scored_names(state)
    if already:
        logger.info("Resuming job %s, skipping %s already-scored", job_id, len(already))

    state["total_files"] = len(applicants)
    jobstore.save(state)

    for applicant in applicants:
        if applicant.name in already:
            continue

        try:
            _resume_text_for(applicant, provider, DEFAULT_MODEL)
            text = build_scoring_text(applicant)
            evaluation = evaluator.evaluate_resume(text)
            record = _result_record(applicant, evaluation=evaluation)
            state["succeeded"] = state.get("succeeded", 0) + 1
        except Exception as e:
            logger.exception("Scoring failed for %s", applicant.name)
            record = _result_record(applicant, error=str(e))
            state["failed"] = state.get("failed", 0) + 1

        state["results"].append(record)
        state["processed_files"] = len(state["results"])

        if record["success"]:
            jobstore.append_result_row(job_id, evaluation_to_csv_row(record))

        # Persisted after every applicant: a crash here keeps everything above.
        jobstore.save(state)

    # Rewrite in ranked order, then publish the export copy. Status flips to
    # completed only after the CSV is on disk, so a client that polls and
    # immediately downloads can never race a missing file.
    ranked = sorted(
        (r for r in state["results"] if r["success"]),
        key=lambda r: (-r["total_score"], r["candidate_name"].lower()),
    )
    if ranked:
        jobstore.rewrite_results_csv(
            job_id, [evaluation_to_csv_row(r) for r in ranked]
        )
        try:
            shutil.copyfile(
                jobstore.results_csv_path(job_id), OUTPUT_DIR / f"{job_id}.csv"
            )
        except OSError as e:
            logger.warning("Could not copy export for job %s: %s", job_id, e)

    state["status"] = jobstore.STATUS_COMPLETED
    jobstore.save(state)
    logger.info(
        "Job %s complete: %s scored, %s failed",
        job_id, state.get("succeeded", 0), state.get("failed", 0),
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    job_id = str(uuid.uuid4())[:8]
    csv_paths: List[str] = []
    zip_paths: List[Path] = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            continue
        dest = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        (zip_paths if suffix == ".zip" else csv_paths).append(
            dest if suffix == ".zip" else str(dest)
        )

    if not csv_paths:
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload the Google Forms CSV export. A ZIP of resumes can be "
                "included alongside it."
            ),
        )

    resumes_dir: Optional[Path] = None
    if zip_paths:
        resumes_dir = UPLOAD_DIR / f"{job_id}_resumes"
        resumes_dir.mkdir(exist_ok=True)
        root = resumes_dir.resolve()
        for zip_path in zip_paths:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    for member in zf.namelist():
                        target = (resumes_dir / member).resolve()
                        if not str(target).startswith(str(root)):
                            continue  # guard against zip-slip path traversal
                        zf.extract(member, resumes_dir)
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{zip_path.name}' is not a readable ZIP archive",
                )

    jobstore.create(job_id, csv_paths, str(resumes_dir) if resumes_dir else None)
    background_tasks.add_task(process_job, job_id, csv_paths, resumes_dir)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Lightweight progress only.

    The full results list is fetched separately, so a 1.5s poll does not
    re-serialize every evaluation on every tick.
    """
    state = jobstore.load(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": state.get("status"),
        "total_files": state.get("total_files", 0),
        "processed_files": state.get("processed_files", 0),
        "succeeded": state.get("succeeded", 0),
        "failed": state.get("failed", 0),
        "error": state.get("error"),
    }


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    if not jobstore.load(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(
        "results.html", {"request": request, "job_id": job_id}
    )


@app.get("/results/{job_id}/data")
async def results_data(job_id: str):
    state = jobstore.load(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


@app.get("/export/{job_id}")
async def export_csv(job_id: str):
    """Serve the results CSV straight from disk.

    Deliberately does not require an in-memory job, so exports remain
    downloadable after a restart.
    """
    for candidate in (jobstore.results_csv_path(job_id), OUTPUT_DIR / f"{job_id}.csv"):
        if candidate.exists():
            return FileResponse(
                candidate,
                media_type="text/csv",
                filename=f"evaluations_{job_id}.csv",
            )
    raise HTTPException(
        status_code=404, detail="No results have been written for this job yet"
    )


@app.get("/jobs")
async def list_jobs():
    return {
        "jobs": [
            {
                "job_id": s.get("job_id"),
                "status": s.get("status"),
                "processed_files": s.get("processed_files", 0),
                "total_files": s.get("total_files", 0),
                "created_at": s.get("created_at"),
            }
            for s in jobstore.list_jobs()
        ]
    }


@app.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, background_tasks: BackgroundTasks):
    """Restart an interrupted job, skipping applicants already scored."""
    state = jobstore.load(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.get("status") == jobstore.STATUS_PROCESSING:
        raise HTTPException(status_code=409, detail="Job is already running")

    resumes_dir = state.get("resumes_dir")
    background_tasks.add_task(
        process_job, job_id, state.get("file_paths", []),
        Path(resumes_dir) if resumes_dir else None,
    )
    return {"job_id": job_id, "status": "resuming"}


@app.get("/rubric", response_class=HTMLResponse)
async def rubric_page(request: Request):
    return templates.TemplateResponse("rubric.html", {"request": request})


@app.post("/rubric/preset/{preset_name}")
async def save_rubric_preset(preset_name: str, config: RubricConfig):
    presets_dir = Path("rubric_presets")
    presets_dir.mkdir(exist_ok=True)
    (presets_dir / f"{preset_name}.json").write_text(
        json.dumps(config.model_dump()), encoding="utf-8"
    )
    return {"message": f"Preset '{preset_name}' saved"}


@app.get("/rubric/presets")
async def list_rubric_presets():
    presets_dir = Path("rubric_presets")
    if not presets_dir.exists():
        return {"presets": []}
    return {"presets": [f.stem for f in presets_dir.glob("*.json")]}


@app.get("/rubric/preset/{preset_name}")
async def load_rubric_preset(preset_name: str):
    preset_path = Path("rubric_presets") / f"{preset_name}.json"
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail="Preset not found")
    return json.loads(preset_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
