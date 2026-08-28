# AI Application Screener

<p align="center"><strong>University club application screening tool</strong> that evaluates candidates for consultant positions using AI-powered analysis of their essay responses, commitments, and resume.</p>

---

## Overview

This tool helps a university club screen applications for consultant candidates. It takes a Google Forms application export (CSV) plus a ZIP of resumes, matches each applicant to their resume by name, evaluates each candidate across four essay/commitment-driven categories, and outputs fair, explainable scores. Bare resume PDFs (without a CSV) can still be scored on resume content alone.

### Evaluation Categories

| Category | Weight | Description |
|----------|--------|-------------|
| Motivation & Fit | 25% | Genuine, specific motivation for joining the club |
| Collaboration & Perspective-Taking | 25% | Working with people whose perspectives differed from their own |
| Values & Judgment | 25% | A value that guides their decisions, and evidence of acting on it |
| Commitments, Leadership & Experience | 25% | Extracurriculars, jobs, volunteering, and resume experience/leadership |

GPA is **not** scored. It's a separate eligibility gate — applicants below 2.5 GPA
are flagged as ineligible in the results, but the number itself never affects the
score above (unparseable GPA values, e.g. "N/A" or a note about transferring, are
flagged for manual review instead of auto-failing).

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Ollama** for local LLM (no API key needed)

### Setup

```bash
git clone <your-repo-url>
cd hiring-agent

python -m venv .venv
source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Ollama Setup

```bash
ollama serve  # Start Ollama
ollama pull gemma3:4b  # Pull the default model
```

---

## Usage

### CLI - Single Resume

```bash
python score.py ./resume/sample.pdf
```

### CLI - Batch Processing

```bash
python score.py ./resumes/
python score.py ./resumes/ ./output/results.csv
```

### Web UI - Google Forms Application (CSV + Resume ZIP)

```bash
python app.py
```

Upload the Google Forms CSV export together with a ZIP of applicant resumes in
one go. Each applicant is matched to their resume by fuzzy name match (the
"Resume" column in the CSV is a Drive link the app can't fetch, so the ZIP's
filenames need to reasonably match the applicant's name). Each CSV row becomes
its own scored row in the results table — it does not need to be one file at a
time.

### Web UI - Bare Resume PDFs

Then open http://localhost:8000 in your browser.

**Features:**
- Drag-and-drop PDF upload
- Real-time processing progress
- Results dashboard with sorting and filtering
- Customizable scoring rubric
- CSV export

---

## Configuration

Copy the template and set your environment variables:

```bash
cp .env.example .env
```

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend (Ollama only) |
| `DEFAULT_MODEL` | `gemma3:4b` | Model name |

---

## Customizing the Rubric

### Via Web UI

1. Go to http://localhost:8000/rubric
2. Adjust category weights using sliders
3. Save custom presets for different hiring rounds

### Via API

```bash
# Save a preset
curl -X POST http://localhost:8000/rubric/preset/my-preset \
  -H "Content-Type: application/json" \
  -d '{"leadership_teamwork": 30, "communication_skills": 25, "problem_solving": 20, "relevant_experience": 15, "academic_performance": 10}'

# List presets
curl http://localhost:8000/rubric/presets

# Load a preset
curl http://localhost:8000/rubric/preset/my-preset
```

---

## Directory Structure

```
.
├── app.py                    # FastAPI web server
├── batch.py                  # Batch processing logic
├── config.py                 # Provider configuration
├── evaluator.py              # Resume evaluation logic
├── llm_utils.py              # LLM provider utilities
├── models.py                 # Pydantic schemas
├── pdf.py                    # PDF extraction
├── prompt.py                 # Prompt utilities
├── prompts/
│   ├── template_manager.py
│   └── templates/            # Evaluation criteria templates
├── pymupdf_rag.py            # PDF to Markdown conversion
├── requirements.txt
├── score.py                  # CLI entry point
├── static/
│   └── style.css             # Web UI styling
├── templates/
│   ├── base.html
│   ├── upload.html
│   ├── results.html
│   └── rubric.html
├── transform.py              # Data transformation
└── resume/
    └── sample.pdf
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Upload page |
| POST | `/upload` | Upload PDF files |
| GET | `/jobs/{job_id}` | Get job status |
| GET | `/results/{job_id}` | Results page |
| GET | `/results/{job_id}/data` | Results JSON data |
| GET | `/export/{job_id}` | Download CSV |
| GET | `/rubric` | Rubric customization page |
| POST | `/rubric/preset/{name}` | Save rubric preset |
| GET | `/rubric/presets` | List saved presets |
| GET | `/rubric/preset/{name}` | Load rubric preset |

---

## License

MIT
