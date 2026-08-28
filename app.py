import os
import json
import uuid
import shutil
import zipfile
import re
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

import csv as csv_mod

from score import main as evaluate_single
from batch import process_batch, write_batch_csv
from models import EvaluationData
from transform import transform_evaluation_response, convert_json_resume_to_text
from pdf import PDFHandler

GPA_THRESHOLD = 2.5

app = FastAPI(title="Resume Screener")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

jobs = {}


class RubricConfig(BaseModel):
    motivation_fit: float = 25
    collaboration_perspective: float = 25
    values_judgment: float = 25
    commitments_experience: float = 25
    gpa_threshold: float = GPA_THRESHOLD


class JobStatus(BaseModel):
    job_id: str
    status: str
    total_files: int = 0
    processed_files: int = 0
    results: List[dict] = []
    error: Optional[str] = None


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _find_column(fieldnames: List[str], keywords: List[str]) -> Optional[str]:
    """Find the CSV header containing all given (lowercase) keywords."""
    for fn in fieldnames or []:
        low = fn.lower()
        if all(k in low for k in keywords):
            return fn
    return None


def _find_resume_for_name(name: str, resumes_dir: Optional[Path]) -> Optional[Path]:
    """Fuzzy-match an applicant name against extracted resume filenames."""
    if not resumes_dir or not name:
        return None
    resumes_dir = Path(resumes_dir)
    if not resumes_dir.exists():
        return None

    target = _normalize_name(name)
    if not target:
        return None

    pdf_files = [p for p in resumes_dir.rglob("*.pdf")]

    for p in pdf_files:
        stem_norm = _normalize_name(p.stem)
        if stem_norm and (stem_norm == target or target in stem_norm or stem_norm in target):
            return p

    name_tokens = [t for t in re.split(r"\s+", name.lower()) if t]
    if name_tokens:
        for p in pdf_files:
            fname_lower = p.stem.lower()
            matches = sum(1 for t in name_tokens if t in fname_lower)
            if matches >= max(1, len(name_tokens) - 1):
                return p

    return None


def _parse_gpa(raw: str):
    """Returns (float_or_None, status) with status in
    eligible / below_gpa_threshold / gpa_unparseable."""
    if not raw or not raw.strip():
        return None, "gpa_unparseable"
    match = re.search(r"\d+\.?\d*", raw)
    if not match:
        return None, "gpa_unparseable"
    try:
        value = float(match.group())
    except ValueError:
        return None, "gpa_unparseable"
    if value < GPA_THRESHOLD:
        return value, "below_gpa_threshold"
    return value, "eligible"


def process_google_forms_csv(file_path: str, resumes_dir: Optional[Path] = None):
    """Yield one result dict per applicant row in a Google Forms export.

    Combines each applicant's essay answers + commitments with their matched
    resume (looked up by fuzzy name match against `resumes_dir`), then scores
    the combination against the essay-based rubric.
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return

    name_col = _find_column(fieldnames, ["name"])
    gpa_col = _find_column(fieldnames, ["gpa"])
    commitments_col = _find_column(fieldnames, ["commitments"])
    circumstances_col = _find_column(fieldnames, ["circumstances"])
    why_col = _find_column(fieldnames, ["interested"]) or _find_column(
        fieldnames, ["join"]
    )
    collab_col = _find_column(fieldnames, ["collaborated"])
    values_col = _find_column(fieldnames, ["value that guides"]) or _find_column(
        fieldnames, ["guides your decisions"]
    )

    from evaluator import ResumeEvaluator
    from prompt import DEFAULT_MODEL, MODEL_PARAMETERS

    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)
    pdf_handler = PDFHandler()

    for row in rows:
        name = (row.get(name_col) or "").strip() if name_col else ""
        candidate_name = name or "Unknown Applicant"

        try:
            gpa_raw = (row.get(gpa_col) or "").strip() if gpa_col else ""
            gpa_value, gpa_status = _parse_gpa(gpa_raw)

            commitments = (row.get(commitments_col) or "").strip() if commitments_col else ""
            circumstances = (
                (row.get(circumstances_col) or "").strip() if circumstances_col else ""
            )
            why_text = (row.get(why_col) or "").strip() if why_col else ""
            collab_text = (row.get(collab_col) or "").strip() if collab_col else ""
            values_text = (row.get(values_col) or "").strip() if values_col else ""

            resume_path = _find_resume_for_name(name, resumes_dir)
            resume_text = ""
            resume_matched = False
            if resume_path:
                try:
                    resume_data = pdf_handler.extract_json_from_pdf(str(resume_path))
                    if resume_data:
                        resume_text = convert_json_resume_to_text(resume_data)
                        resume_matched = True
                except Exception:
                    resume_text = ""

            text_parts = [
                f"Why interested essay:\n{why_text or 'Not provided'}",
                f"\nCollaboration essay:\n{collab_text or 'Not provided'}",
                f"\nValues essay:\n{values_text or 'Not provided'}",
                f"\nCurrent commitments:\n{commitments or 'Not provided'}",
            ]
            if circumstances:
                text_parts.append(f"\nCircumstances noted by applicant:\n{circumstances}")
            if resume_text:
                text_parts.append(f"\nResume:\n{resume_text}")

            combined_text = "\n".join(text_parts)

            evaluation = evaluator.evaluate_resume(combined_text)
            if evaluation:
                evaluation.gpa_status = gpa_status
                evaluation.gpa_value = gpa_raw

            yield {
                "file_name": Path(file_path).name,
                "candidate_name": candidate_name,
                "evaluation": evaluation.model_dump() if evaluation else None,
                "success": evaluation is not None,
                "resume_matched": resume_matched,
            }
        except Exception as e:
            yield {
                "file_name": Path(file_path).name,
                "candidate_name": candidate_name,
                "evaluation": None,
                "success": False,
                "error": str(e),
            }


def process_job(job_id: str, file_paths: List[str], resumes_dir: Optional[str] = None):
    try:
        jobs[job_id]["status"] = "processing"
        results = []

        # Google Forms CSVs expand into one result per applicant row, so total
        # progress must count rows, not just uploaded files.
        total_units = 0
        for file_path in file_paths:
            if Path(file_path).suffix.lower() == ".csv":
                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        total_units += sum(1 for _ in csv_mod.DictReader(f))
                except Exception:
                    pass
            else:
                total_units += 1
        jobs[job_id]["total_files"] = total_units

        processed = 0
        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()

            if ext == ".csv":
                for candidate_result in process_google_forms_csv(file_path, resumes_dir):
                    results.append(candidate_result)
                    processed += 1
                    jobs[job_id]["processed_files"] = processed
                    jobs[job_id]["results"] = results
                continue

            try:
                result = evaluate_single(file_path)
                if result:
                    results.append({
                        "file_name": Path(file_path).name,
                        "candidate_name": Path(file_path).stem,
                        "evaluation": result.model_dump() if result else None,
                        "success": True,
                    })
                else:
                    results.append({
                        "file_name": Path(file_path).name,
                        "candidate_name": Path(file_path).stem,
                        "evaluation": None,
                        "success": False,
                        "error": "Evaluation returned None",
                    })
            except Exception as e:
                results.append({
                    "file_name": Path(file_path).name,
                    "candidate_name": Path(file_path).stem,
                    "evaluation": None,
                    "success": False,
                    "error": str(e),
                })

            processed += 1
            jobs[job_id]["processed_files"] = processed
            jobs[job_id]["results"] = results

        jobs[job_id]["status"] = "completed"

        csv_path = OUTPUT_DIR / f"{job_id}.csv"
        successful_results = [r for r in results if r.get("success") and r.get("evaluation")]
        if successful_results:
            csv_rows = []
            for r in successful_results:
                eval_data = EvaluationData(**r["evaluation"])
                row = transform_evaluation_response(
                    file_name=r["file_name"],
                    evaluation=eval_data,
                )
                csv_rows.append(row)

            if csv_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = list(csv_rows[0].keys())
                    writer = csv_mod.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    job_id = str(uuid.uuid4())[:8]
    file_paths = []
    zip_paths = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in (".pdf", ".csv", ".zip"):
            continue

        file_path = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if ext == ".zip":
            zip_paths.append(file_path)
        else:
            file_paths.append(str(file_path))

    if not file_paths and not zip_paths:
        raise HTTPException(status_code=400, detail="No valid PDF, CSV, or ZIP files uploaded")
    if not file_paths:
        raise HTTPException(
            status_code=400, detail="A ZIP of resumes was uploaded without a CSV to match it against"
        )

    resumes_dir: Optional[Path] = None
    if zip_paths:
        resumes_dir = UPLOAD_DIR / f"{job_id}_resumes"
        resumes_dir.mkdir(exist_ok=True)
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    member_path = (resumes_dir / member).resolve()
                    if not str(member_path).startswith(str(resumes_dir.resolve())):
                        continue  # guard against zip-slip path traversal
                    zf.extract(member, resumes_dir)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total_files": len(file_paths),
        "processed_files": 0,
        "results": [],
        "file_paths": file_paths,
        "resumes_dir": str(resumes_dir) if resumes_dir else None,
    }

    if background_tasks:
        background_tasks.add_task(process_job, job_id, file_paths, resumes_dir)
    else:
        process_job(job_id, file_paths, resumes_dir)

    return {"job_id": job_id, "total_files": len(file_paths)}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse("results.html", {"request": request, "job_id": job_id, "job": jobs[job_id]})


@app.get("/results/{job_id}/data")
async def results_data(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/export/{job_id}")
async def export_csv(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    csv_path = OUTPUT_DIR / f"{job_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not found")

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"evaluations_{job_id}.csv",
    )


@app.get("/rubric", response_class=HTMLResponse)
async def rubric_page(request: Request):
    return templates.TemplateResponse("rubric.html", {"request": request})


@app.post("/rubric/preset/{preset_name}")
async def save_rubric_preset(preset_name: str, config: RubricConfig):
    presets_dir = Path("rubric_presets")
    presets_dir.mkdir(exist_ok=True)

    preset_path = presets_dir / f"{preset_name}.json"
    with open(preset_path, "w") as f:
        json.dump(config.model_dump(), f)

    return {"message": f"Preset '{preset_name}' saved"}


@app.get("/rubric/presets")
async def list_rubric_presets():
    presets_dir = Path("rubric_presets")
    if not presets_dir.exists():
        return {"presets": []}

    presets = [f.stem for f in presets_dir.glob("*.json")]
    return {"presets": presets}


@app.get("/rubric/preset/{preset_name}")
async def load_rubric_preset(preset_name: str):
    presets_dir = Path("rubric_presets")
    preset_path = presets_dir / f"{preset_name}.json"

    if not preset_path.exists():
        raise HTTPException(status_code=404, detail="Preset not found")

    with open(preset_path) as f:
        config = json.load(f)

    return config


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
