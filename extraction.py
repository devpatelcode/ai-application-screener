"""Resume text extraction for the formats applicants actually submit.

Google Forms accepts whatever the applicant uploads, so a real batch is a mix of
PDF, DOCX and image files. Every supported format is reduced to a single plain
text blob that goes straight into the scoring prompt -- there is deliberately no
intermediate "parse into JSON Resume" step, which used to cost 6-12 extra LLM
calls per applicant and discarded the whole resume if any one section failed.

Extraction never fails silently: an unreadable resume returns an explicit reason
so the applicant can be flagged rather than quietly scored without it.
"""

import base64
import logging
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf
from pymupdf_rag import to_markdown

logger = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | DOCX_SUFFIXES | IMAGE_SUFFIXES

# Below this, extraction "succeeded" but produced nothing usable (e.g. a scanned
# PDF with no text layer). Treated as a failure so the applicant gets flagged.
MIN_USABLE_CHARS = 50

VISION_PROMPT = (
    "This image is a resume. Transcribe all of its text content verbatim as "
    "plain text, preserving the section order and structure. Do not summarize, "
    "evaluate, or add commentary."
)


@dataclass
class ExtractResult:
    """Outcome of reading one resume file."""

    text: str = ""
    method: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def _extract_pdf(path: Path) -> ExtractResult:
    """PDF -> markdown via pymupdf4llm. No LLM call."""
    try:
        with pymupdf.open(path) as doc:
            text = to_markdown(doc, pages=range(doc.page_count)) or ""
    except Exception as e:
        return ExtractResult(method="pdf", error=f"PDF could not be read: {e}")
    return ExtractResult(text=text.strip(), method="pdf")


def _docx_via_textutil(path: Path) -> Optional[str]:
    """macOS ships textutil, which handles .docx formatting better than raw XML."""
    if not shutil.which("textutil"):
        return None
    try:
        out = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("textutil failed on %s: %s", path.name, e)
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", errors="replace").strip() or None


def _docx_via_stdlib(path: Path) -> Optional[str]:
    """Fallback with no third-party dependency: a .docx is a zip of XML.

    Paragraph and row boundaries are converted to newlines before tags are
    stripped, so the text keeps its line structure instead of collapsing into
    one run-on blob.
    """
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        logger.debug("stdlib docx read failed on %s: %s", path.name, e)
        return None

    xml = re.sub(r"</w:(p|tr)>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip() or None


def _docx_is_empty(path: Path) -> bool:
    """True when the document genuinely contains no text runs.

    Distinguishes "the applicant uploaded a blank file" from "we failed to parse
    it" -- only the first is actionable by asking for a resubmission.
    """
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
        has_media = any("word/media/" in n for n in zipfile.ZipFile(path).namelist())
        return "<w:t" not in xml and not has_media
    except Exception:
        return False


def _extract_docx(path: Path) -> ExtractResult:
    text = _docx_via_textutil(path)
    method = "docx:textutil"
    if not text:
        text = _docx_via_stdlib(path)
        method = "docx:stdlib"
    if not text:
        if _docx_is_empty(path):
            return ExtractResult(
                method="docx",
                error="The uploaded document is blank (no text) — ask the applicant to resubmit",
            )
        return ExtractResult(
            method="docx", error="DOCX could not be read by textutil or stdlib"
        )
    return ExtractResult(text=text, method=method)


def _extract_image(path: Path, provider=None, model: Optional[str] = None) -> ExtractResult:
    """Image -> text by asking a vision-capable model to transcribe it.

    This is the only extraction path that costs an LLM call. It is used for the
    small number of applicants who upload a screenshot instead of a document.
    """
    if provider is None or not model:
        return ExtractResult(
            method="image",
            error="Image resume requires a vision-capable model; none configured",
        )
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        response = provider.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        },
                    ],
                }
            ],
            options={"temperature": 0.0, "top_p": 0.9},
        )
        text = (response["message"]["content"] or "").strip()
    except Exception as e:
        return ExtractResult(method="image", error=f"Image transcription failed: {e}")

    if not text:
        return ExtractResult(method="image", error="Image transcription returned nothing")
    return ExtractResult(text=text, method="image:vision")


def extract_resume_text(
    path, provider=None, vision_model: Optional[str] = None
) -> ExtractResult:
    """Read a resume of any supported format into plain text.

    `provider`/`vision_model` are only needed for image resumes; PDF and DOCX
    extraction is entirely local.
    """
    path = Path(path)
    if not path.exists():
        return ExtractResult(error=f"File not found: {path.name}")

    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        result = _extract_pdf(path)
    elif suffix in DOCX_SUFFIXES:
        result = _extract_docx(path)
    elif suffix in IMAGE_SUFFIXES:
        result = _extract_image(path, provider=provider, model=vision_model)
    elif suffix == ".doc":
        # Legacy binary .doc; textutil can still handle it on macOS.
        text = _docx_via_textutil(path)
        result = (
            ExtractResult(text=text, method="doc:textutil")
            if text
            else ExtractResult(method="doc", error="Legacy .doc could not be read")
        )
    else:
        return ExtractResult(
            method=suffix.lstrip(".") or "unknown",
            error=f"Unsupported resume format '{suffix or 'none'}'",
        )

    if result.error:
        return result
    if len(result.text) < MIN_USABLE_CHARS:
        return ExtractResult(
            method=result.method,
            error=(
                f"Extracted only {len(result.text)} characters "
                "(likely a scanned document with no text layer)"
            ),
        )
    return result
