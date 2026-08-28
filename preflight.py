"""Guard against the failure that produces wrong scores with no error.

Ollama defaults to a 4096-token context and silently truncates anything longer.
Truncation removes the *end* of the prompt -- which is exactly where the
applicant's essays and resume sit -- so the model scores whatever survived and
still returns perfectly well-formed JSON. There is no exception, no warning, and
no way to tell a truncated score from a real one after the fact.

The only safe posture is to measure the real serving window up front and refuse
to run when the prompt will not fit.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Rough chars-per-token for English prose. Deliberately conservative (real
# ratio is ~4) so the estimate errs toward "too big" rather than too small.
CHARS_PER_TOKEN = 3.5

# Headroom for the model's own JSON output plus chat formatting overhead.
RESPONSE_TOKEN_RESERVE = 900

RECOMMENDED_CONTEXT = 32768


@dataclass
class ContextInfo:
    """What the backend is actually serving, as opposed to what we asked for."""

    context_length: Optional[int]
    model: Optional[str] = None
    source: str = "unknown"

    @property
    def known(self) -> bool:
        return bool(self.context_length)


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def _ollama_host(base_url: str) -> Optional[str]:
    """Map an Ollama OpenAI-compat base_url to its native API root."""
    if not base_url:
        return None
    root = base_url.rstrip("/")
    for suffix in ("/v1", "/v1beta/openai"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    else:
        return None
    return root if "localhost" in root or "127.0.0.1" in root else None


def probe_context(base_url: str, model: Optional[str] = None) -> ContextInfo:
    """Ask Ollama what context window the loaded model is actually serving.

    Returns an unknown result for remote providers, where the window is a
    published property of the model rather than something we can measure.
    """
    host = _ollama_host(base_url)
    if not host:
        return ContextInfo(context_length=None, model=model, source="remote")

    try:
        resp = requests.get(f"{host}/api/ps", timeout=5)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            if not model or entry.get("name", "").startswith(model.split(":")[0]):
                return ContextInfo(
                    context_length=entry.get("context_length"),
                    model=entry.get("name"),
                    source="ollama:/api/ps",
                )
    except (requests.RequestException, ValueError) as e:
        logger.debug("Could not probe Ollama context: %s", e)

    # Nothing loaded yet: fall back to the server-wide default, which is what a
    # freshly loaded model will use.
    env = os.getenv("OLLAMA_CONTEXT_LENGTH")
    if env and env.isdigit():
        return ContextInfo(int(env), model, "env:OLLAMA_CONTEXT_LENGTH")
    return ContextInfo(None, model, "ollama:not-loaded")


def remediation(required_tokens: int) -> str:
    target = max(RECOMMENDED_CONTEXT, 1 << (max(required_tokens, 1) - 1).bit_length())
    return (
        "The model's context window is too small for these applications, and "
        "Ollama truncates silently -- which produces confident but WRONG scores.\n"
        f"Restart Ollama with a larger window:\n\n"
        f"    OLLAMA_CONTEXT_LENGTH={target} ollama serve\n\n"
        "Per-request num_ctx does not override this; it must be set when the "
        "server starts."
    )


def check_prompt_fits(prompt_chars: int, ctx: ContextInfo, label: str = "prompt"):
    """Return (ok, message) for a prompt of the given size.

    An unknown window is allowed through with a warning rather than blocking a
    remote provider we cannot measure.
    """
    needed = estimate_tokens(" " * prompt_chars) + RESPONSE_TOKEN_RESERVE

    if not ctx.known:
        return True, (
            f"Could not determine the model's context window ({ctx.source}); "
            f"proceeding, but {label} needs roughly {needed} tokens."
        )

    if needed > ctx.context_length:
        return False, (
            f"{label} needs about {needed} tokens but the model is serving only "
            f"{ctx.context_length} ({ctx.source}).\n\n" + remediation(needed)
        )

    return True, (
        f"Context OK: {label} needs about {needed} tokens, "
        f"model is serving {ctx.context_length}."
    )
