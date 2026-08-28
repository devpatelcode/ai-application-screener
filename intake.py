"""Turn a Google Forms export (+ the resume ZIP) into scoreable applicant records.

Three things here are deliberately strict, because each one previously failed
silently and produced plausible-looking but wrong scores:

* Required columns must resolve, or intake raises before any scoring starts.
  A reworded form question used to yield "Not provided" for every applicant,
  zeroing an entire 25-point category across the whole batch with no warning.
* A resume is attached only on an unambiguous name match. The old substring
  heuristic could attach another applicant's resume.
* GPA is a pass/fail eligibility gate only, never a score input, and anything
  it cannot confidently read is routed to human review rather than rejected.
"""

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from extraction import SUPPORTED_SUFFIXES

logger = logging.getLogger(__name__)

GPA_THRESHOLD = 2.5
GPA_MAX_PLAUSIBLE = 5.0

# Resume match / eligibility states surfaced to reviewers.
RESUME_MATCHED = "matched"
RESUME_MISSING = "missing"
RESUME_UNREADABLE = "unreadable"
RESUME_AMBIGUOUS = "ambiguous"

GPA_ELIGIBLE = "eligible"
GPA_BELOW_THRESHOLD = "below_gpa_threshold"
GPA_NEEDS_REVIEW = "needs_review"

# Phrases indicating the applicant has no GPA yet. Checked before numeric
# parsing so "0.00 (no grades yet)" is not read as a literal 0.0.
_NO_GRADES_MARKERS = (
    "no grade",
    "not available",
    "unavailable",
    "first semester",
    "first-year",
    "freshmen",
    "freshman",
    "incoming",
    "transfer",
    "pending",
    "n/a",
    "na since",
    "tbd",
)


@dataclass
class ColumnMap:
    """Resolved CSV headers for the fields intake needs."""

    name: str
    why: str
    collaboration: str
    values: str
    gpa: Optional[str] = None
    commitments: Optional[str] = None
    circumstances: Optional[str] = None
    email: Optional[str] = None
    year: Optional[str] = None


@dataclass
class Applicant:
    """One form response, plus whatever resume we could attach to it."""

    row_index: int
    name: str
    email: str = ""
    year: str = ""
    why: str = ""
    collaboration: str = ""
    values: str = ""
    commitments: str = ""
    circumstances: str = ""
    gpa_raw: str = ""
    gpa_value: Optional[float] = None
    gpa_status: str = GPA_NEEDS_REVIEW
    resume_path: Optional[Path] = None
    resume_status: str = RESUME_MISSING
    resume_note: str = ""
    resume_text: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def ineligible(self) -> bool:
        return self.gpa_status == GPA_BELOW_THRESHOLD


class IntakeError(Exception):
    """Raised when the CSV cannot be interpreted at all."""


def normalize_name(s: str) -> str:
    """Casefold, strip accents and punctuation, for name comparison."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_column(fieldnames: List[str], keywords: List[str]) -> Optional[str]:
    """First header containing every keyword (all lowercase substrings)."""
    for fn in fieldnames or []:
        low = fn.lower()
        if all(k in low for k in keywords):
            return fn
    return None


def resolve_columns(fieldnames: List[str]) -> ColumnMap:
    """Map form questions to CSV headers, failing loudly on required ones."""
    name = find_column(fieldnames, ["name"])
    why = find_column(fieldnames, ["interested"]) or find_column(
        fieldnames, ["why", "join"]
    )
    collaboration = find_column(fieldnames, ["collaborated"]) or find_column(
        fieldnames, ["perspectives"]
    )
    values = find_column(fieldnames, ["value that guides"]) or find_column(
        fieldnames, ["guides your decisions"]
    )

    missing = [
        label
        for label, col in (
            ("applicant name", name),
            ("'why are you interested' essay", why),
            ("'collaborated with differing perspectives' essay", collaboration),
            ("'value that guides your decisions' essay", values),
        )
        if not col
    ]
    if missing:
        raise IntakeError(
            "Could not find required column(s) in the CSV: "
            + ", ".join(missing)
            + ".\nHeaders present: "
            + ", ".join(repr(f) for f in (fieldnames or []))
            + "\nIf the form wording changed, update the keywords in "
            "intake.resolve_columns()."
        )

    return ColumnMap(
        name=name,
        why=why,
        collaboration=collaboration,
        values=values,
        gpa=find_column(fieldnames, ["gpa"]),
        commitments=find_column(fieldnames, ["commitments"]),
        circumstances=find_column(fieldnames, ["circumstances"]),
        email=find_column(fieldnames, ["email"]),
        year=find_column(fieldnames, ["year"]),
    )


def parse_gpa(raw: str):
    """Return (value, status) for a self-reported GPA.

    Only a confidently parsed number below the threshold rejects anyone.
    Blank, non-numeric, no-grades-yet, implausible-scale and literal-zero
    values all go to human review instead, since a real 0.0 is indistinguishable
    from "I have no grades yet".
    """
    text = (raw or "").strip()
    if not text:
        return None, GPA_NEEDS_REVIEW

    lowered = text.lower()
    if any(marker in lowered for marker in _NO_GRADES_MARKERS):
        return None, GPA_NEEDS_REVIEW

    # Decimal-bearing numbers first (a real GPA is almost always written "3.4"),
    # falling back to a bare integer like "4".
    matches = re.findall(r"\d+\.\d+", text) or re.findall(r"\b\d\b", text)
    if not matches:
        return None, GPA_NEEDS_REVIEW

    try:
        value = float(matches[0])
    except ValueError:
        return None, GPA_NEEDS_REVIEW

    # Out-of-scale (e.g. a 10- or 100-point scale) or a literal zero: don't guess.
    if value <= 0.0 or value > GPA_MAX_PLAUSIBLE:
        return value, GPA_NEEDS_REVIEW

    # More than one distinct number ("3.4 weighted, 3.2 unweighted") is ambiguous.
    distinct = {m for m in matches}
    if len(distinct) > 1:
        return value, GPA_NEEDS_REVIEW

    if value < GPA_THRESHOLD:
        return value, GPA_BELOW_THRESHOLD
    return value, GPA_ELIGIBLE


def index_resumes(resumes_dir) -> Dict[str, List[Path]]:
    """Index resume files by the owner name Google Forms appends to the filename.

    Forms names uploads "<original name> - <Full Name>.<ext>", so the text after
    the last " - " is the applicant's Drive display name -- a far more reliable
    key than fuzzy-matching the whole filename.
    """
    index: Dict[str, List[Path]] = {}
    if not resumes_dir:
        return index
    resumes_dir = Path(resumes_dir)
    if not resumes_dir.exists():
        return index

    for path in sorted(resumes_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES and path.suffix.lower() != ".doc":
            continue
        owner = path.stem.rsplit(" - ", 1)[-1] if " - " in path.stem else path.stem
        key = normalize_name(owner)
        if key:
            index.setdefault(key, []).append(path)
    return index


def match_resume(name: str, index: Dict[str, List[Path]]):
    """Find this applicant's resume. Returns (path, status, note).

    Requires a unique match; ambiguity is reported rather than guessed at.
    """
    target = normalize_name(name)
    if not target or not index:
        return None, RESUME_MISSING, "No resume file matched this applicant"

    exact = index.get(target)
    if exact:
        if len(exact) == 1:
            return exact[0], RESUME_MATCHED, ""
        return (
            None,
            RESUME_AMBIGUOUS,
            f"{len(exact)} resumes share this name: "
            + ", ".join(p.name for p in exact),
        )

    # Handle "Juan Elizondo" (form) vs "Juan Elizondo-Alvarez" (Drive account).
    candidates = [
        p
        for key, paths in index.items()
        if key.startswith(target) or target.startswith(key)
        for p in paths
    ]
    if len(candidates) == 1:
        return candidates[0], RESUME_MATCHED, ""
    if len(candidates) > 1:
        return (
            None,
            RESUME_AMBIGUOUS,
            f"Name is ambiguous across {len(candidates)} files: "
            + ", ".join(p.name for p in candidates),
        )

    return None, RESUME_MISSING, "No resume file matched this applicant"


def load_applicants(csv_path, resumes_dir=None) -> List[Applicant]:
    """Read the form export and pair each response with its resume file.

    Resume *text* is not extracted here -- only located -- so that intake stays
    fast and fully offline; extraction happens per-applicant during scoring.
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = resolve_columns(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise IntakeError("CSV contains a header but no application rows.")

    index = index_resumes(resumes_dir)
    applicants: List[Applicant] = []

    def cell(row, col):
        return (row.get(col) or "").strip() if col else ""

    for i, row in enumerate(rows):
        name = cell(row, columns.name)
        applicant = Applicant(
            row_index=i,
            name=name or f"Unnamed applicant (row {i + 2})",
            email=cell(row, columns.email),
            year=cell(row, columns.year),
            why=cell(row, columns.why),
            collaboration=cell(row, columns.collaboration),
            values=cell(row, columns.values),
            commitments=cell(row, columns.commitments),
            circumstances=cell(row, columns.circumstances),
            gpa_raw=cell(row, columns.gpa),
        )
        applicant.gpa_value, applicant.gpa_status = parse_gpa(applicant.gpa_raw)

        if not name:
            applicant.warnings.append("Name column was empty for this row")

        for label, value in (
            ("why-interested", applicant.why),
            ("collaboration", applicant.collaboration),
            ("values", applicant.values),
        ):
            if not value:
                applicant.warnings.append(f"Blank {label} essay")

        if index:
            path, status, note = match_resume(name, index)
            applicant.resume_path = path
            applicant.resume_status = status
            applicant.resume_note = note
        else:
            applicant.resume_status = RESUME_MISSING
            applicant.resume_note = "No resume archive was uploaded"

        applicants.append(applicant)

    return applicants


def build_scoring_text(applicant: Applicant) -> str:
    """Assemble the applicant-specific half of the scoring prompt."""
    parts = [
        f"Why interested essay:\n{applicant.why or 'Not provided'}",
        f"\nCollaboration essay:\n{applicant.collaboration or 'Not provided'}",
        f"\nValues essay:\n{applicant.values or 'Not provided'}",
        f"\nCurrent commitments:\n{applicant.commitments or 'Not provided'}",
    ]
    if applicant.circumstances:
        parts.append(f"\nCircumstances noted by applicant:\n{applicant.circumstances}")
    if applicant.resume_text:
        parts.append(f"\nResume:\n{applicant.resume_text}")
    else:
        parts.append("\nResume:\nNot provided")
    return "\n".join(parts)
