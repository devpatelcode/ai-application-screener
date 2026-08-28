from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from models import JSONResume, EvaluationData
from llm_utils import initialize_llm_provider, extract_json_from_response
import logging
import json
import re

# One extra attempt when the model returns unparseable or invalid JSON. Small
# models fail this way intermittently, and without a retry a single malformed
# token permanently dropped an applicant from the results.
MAX_PARSE_ATTEMPTS = 2

RETRY_NUDGE = (
    "Your previous response could not be parsed. Respond with ONLY the JSON "
    "object described above -- no explanation, no markdown fence, no reasoning."
)

from prompt import (
    DEFAULT_MODEL,
    MODEL_PARAMETERS,
)
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class ResumeEvaluator:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_params: dict = None):
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.model_name = model_name
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.5, "top_p": 0.9}
        )
        self.template_manager = TemplateManager()
        self._initialize_llm_provider()

    def _initialize_llm_provider(self):
        """Initialize the appropriate LLM provider based on the model."""
        self.provider = initialize_llm_provider(self.model_name)

    def _load_evaluation_prompt(self, resume_text: str) -> str:
        criteria_template = self.template_manager.render_template(
            "resume_evaluation_criteria", text_content=resume_text
        )
        if criteria_template is None:
            raise ValueError("Failed to load resume evaluation criteria template")
        return criteria_template

    def evaluate_resume(self, resume_text: str) -> EvaluationData:
        """Score one application, retrying once if the model's JSON is unusable.

        Transport errors (connection dropped, timeout, 5xx) are retried inside
        the provider; this retry covers the separate case of a reachable model
        returning something that will not parse or validate.
        """
        self._last_resume_text = resume_text
        full_prompt = self._load_evaluation_prompt(resume_text)

        system_message = self.template_manager.render_template(
            "resume_evaluation_system_message"
        )
        if system_message is None:
            raise ValueError("Failed to load resume evaluation system message template")

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": full_prompt},
            ]
            if attempt > 1:
                messages.append({"role": "system", "content": RETRY_NUDGE})

            response = self.provider.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "stream": False,
                    "temperature": self.model_params.get("temperature", 0.1),
                    "top_p": self.model_params.get("top_p", 0.9),
                },
                format=EvaluationData.model_json_schema(),
            )

            raw = response["message"]["content"]
            cleaned = extract_json_from_response(raw)
            logger.debug("Evaluation response (attempt %s): %s", attempt, cleaned)

            try:
                return EvaluationData(**json.loads(cleaned))
            except Exception as e:
                last_error = e
                logger.warning(
                    "Evaluation response invalid on attempt %s/%s: %s",
                    attempt,
                    MAX_PARSE_ATTEMPTS,
                    e,
                )
                if attempt == MAX_PARSE_ATTEMPTS:
                    logger.error("Unparseable model output was: %r", raw[:500])

        raise ValueError(
            f"Model returned unusable JSON after {MAX_PARSE_ATTEMPTS} attempts: {last_error}"
        )
