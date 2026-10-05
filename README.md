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
git clone https://github.com/devpatelcode/ai-application-screener.git
cd ai-application-screener

python -m venv .venv
source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Ollama Setup

> **Important:** Ollama defaults to a 4096-token context window and **truncates
> longer prompts silently**. A truncated prompt loses the end — which is exactly
> where the applicant's essays and resume are — so the model returns a confident,
> well-formed, and *wrong* score with no error. Always start Ollama with a larger
> window:

```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve   # required, not optional
ollama pull gemma3:12b                     # the default model (~8GB)
```

Per-request `num_ctx` does **not** override this — it must be set when the server
starts. The app probes the real window on startup, logs it, and warns loudly if
it is too small, so you never have to take this on faith:

```
Model gemma3:12b serving 32768-token context (ollama:/api/ps)
```

Verify independently with `curl -s http://localhost:11434/api/ps`.

---

## Usage

### Web UI — the normal path (CSV + resume ZIP)

```bash
python app.py
```

Open http://localhost:8000 and upload two files together:

1. the **Google Forms CSV export** of application responses, and
2. the **ZIP of resumes** downloaded from the form's Drive folder.

Each CSV row becomes its own scored candidate. Resumes are matched to applicants
using the owner name Google Forms appends to each uploaded filename
(`<original> - <Full Name>.<ext>`), which is exact rather than a fuzzy guess. A
resume is attached only on an unambiguous match; if two files could belong to the
same person, the applicant is flagged instead of being given a coin-flip resume.

**Supported resume formats:** PDF, DOCX, and images (PNG/JPG, transcribed with a
vision model). Anything unreadable is reported per-candidate rather than silently
skipped, so you always know who was scored without a resume.

**Features:**
- Drag-and-drop upload of CSV + ZIP
- Live progress, and results visible while the run is still going
- Ranked dashboard with search, sorting and per-category evidence
- Resume and GPA status flags for anything needing human review
- CSV export, downloadable at any point during or after the run

### Resuming an interrupted run

Job state is written to `jobs/<job_id>/` after **every** applicant, so nothing is
lost if the server stops mid-run. On restart, unfinished jobs are marked
`interrupted` rather than appearing to hang, and:

```bash
curl -X POST http://localhost:8000/jobs/<job_id>/resume   # skips already-scored applicants
curl http://localhost:8000/jobs                           # list all past jobs
curl -O http://localhost:8000/export/<job_id>             # works even after a restart
```

### CLI — scoring a single resume file

```bash
python score.py ./resume/sample.pdf
```

Useful for spot-checking extraction. Note there are no essay answers on this
path, so the three essay categories score from resume content alone — the web
CSV+ZIP flow is the intended way to review real applications.

---

## Configuration

Copy the template and set your environment variables:

```bash
cp .env.example .env
```

| Variable | Values | Description |
|----------|--------|-------------|
| `DEFAULT_MODEL` | `gemma3:12b` | Model name; must exist in `providers.json`. `gemma3:4b` is ~4x faster but scores far less discriminatingly (see below) |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | **Set on the Ollama server**, not the app. Too small = silent truncation = wrong scores |
| `GEMINI_API_KEY` | — | Only needed if you select a `gemini-*` model |

Applicant data (`uploads/`, `output/`, `jobs/`) is gitignored — it contains real
names, emails, phone numbers, GPAs, essays and resumes. Keep it that way.

### Why `gemma3:12b` and not `gemma3:4b`

Both models were run over the same 45-applicant pool under identical conditions:

| | `gemma3:4b` | `gemma3:12b` |
|---|---|---|
| Distinct scores across 45 applicants | 17 | **26** |
| Applicants sitting in a tie group | 87% | **62%** |
| Largest single tie | 10 people | **6 people** |
| Score spread (stdev) | 5.8 | **14.5** |
| Range used | 67–91 | **39–96** |
| Seconds per applicant | ~13 | ~57 |

Rank correlation between the two is **0.62** — they produce genuinely different
orderings, not the same ordering rescaled. Spot-checking the disagreements against
the rubric showed the 4b was the one getting it wrong, in both directions: it gave
22/25 for motivation to an essay whose only concrete reason was something seen on
the club's social media, and 18/25 for values to an applicant with strong, direct
evidence of acting on the value they named.

The 4b compresses everyone into 67–91, which cannot support a cut decision. Use it
only for smoke-testing the pipeline; use the 12b for real review.

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
  -d '{"motivation_fit": 30, "collaboration_perspective": 25, "values_judgment": 25, "commitments_experience": 20, "gpa_threshold": 2.5}'

# List presets
curl http://localhost:8000/rubric/presets

# Load a preset
curl http://localhost:8000/rubric/preset/my-preset
```

---

## Directory Structure

```
.
├── app.py                    # FastAPI server: HTTP + job orchestration only
├── intake.py                 # Form CSV -> applicants; columns, GPA, resume matching
├── extraction.py             # Resume file -> text (PDF / DOCX / image)
├── evaluator.py              # One scoring call, with JSON-repair retry
├── scoring.py                # THE definition of a candidate's total score
├── jobstore.py               # Durable job state + incremental CSV
├── preflight.py              # Context-window guard against silent truncation
├── models.py                 # Pydantic schemas + LLM transport (retries)
├── config.py                 # Provider/model resolution from providers.json
├── llm_utils.py              # Provider factory + tolerant JSON extraction
├── transform.py              # Result record -> CSV row
├── prompts/templates/        # Rubric + system message (Jinja)
├── templates/                # upload / results / rubric pages
├── static/style.css
├── score.py, batch.py        # CLI entry points
└── jobs/<job_id>/            # Persisted state + results (gitignored)
```

Each module has one job, so a change to the rubric, the file formats, or the
persistence model touches exactly one file.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Upload page |
| POST | `/upload` | Upload the application CSV (+ optional resume ZIP) |
| GET | `/jobs` | List all jobs, including past runs |
| GET | `/jobs/{job_id}` | Lightweight progress for polling |
| POST | `/jobs/{job_id}/resume` | Resume an interrupted job |
| GET | `/results/{job_id}` | Results page |
| GET | `/results/{job_id}/data` | Full results JSON |
| GET | `/export/{job_id}` | Download CSV (served from disk, survives restarts) |
| GET | `/rubric` | Rubric customization page |
| POST | `/rubric/preset/{name}` | Save rubric preset |
| GET | `/rubric/presets` | List saved presets |
| GET | `/rubric/preset/{name}` | Load rubric preset |

---

## Credits

Built on [HackerRank's open-source hiring-agent](https://github.com/interviewstreet/hiring-agent)
(MIT), which provided the original resume-scoring CLI and LLM provider layer. This project
adapts it for club recruiting: the FastAPI web app, Google Forms CSV + resume ZIP intake,
the four-category application rubric, multi-format resume extraction, resumable job state,
the context-window preflight, and the model benchmark above.

## License

MIT. See [LICENSE](LICENSE); the original HackerRank copyright notice is retained.