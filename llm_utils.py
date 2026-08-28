"""
Utility functions for LLM providers.
"""

import logging
from typing import Any, Dict, Optional
from config import provider_for
from models import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


import re

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_from_response(response_text: str) -> str:
    """Pull the JSON object out of whatever the model actually returned.

    Small models wrap their output in unpredictable ways -- bare ``` fences,
    a "Here is the JSON:" preamble, trailing pleasantries, or a reasoning block
    that was cut off before its closing tag. The previous implementation only
    handled an exact ```json prefix with a trailing fence and nothing else, so
    most of those shapes caused the applicant to be dropped entirely.

    Strategy: strip reasoning, prefer fenced content, then fall back to the
    outermost brace pair -- the same salvage the PDF parser already used.
    """
    if not response_text:
        return ""

    text = response_text.strip()

    # Drop a <think> block whether or not it was closed. An unterminated block
    # means the reasoning ran to the end, so there is no JSON after it anyway.
    if "<think>" in text:
        start = text.find("<think>")
        end = text.find("</think>")
        text = (
            text[:start] + text[end + len("</think>") :] if end != -1 else text[:start]
        ).strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # An unclosed fence: keep everything after the opening marker.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        if text.endswith("```"):
            text = text[:-3].rstrip()

    # Final salvage: take the outermost {...}, which discards any surrounding
    # prose the model added before or after the object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return text.strip()


def initialize_llm_provider(model_name: str) -> Any:
    """
    Initialize an OpenAI-compatible LLM provider for the given model,
    resolving base_url / api_key / structured-output mode from providers.json.
    """
    cfg = provider_for(model_name)
    logger.info(f"🔄 Using model {model_name} via {cfg['base_url']}")
    return OpenAICompatibleProvider(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        structured_output=cfg["structured_output"],
        extra_body=cfg["extra_body"],
    )
