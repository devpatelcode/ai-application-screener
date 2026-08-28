from typing import List, Optional, Dict, Tuple, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field, field_validator, model_validator


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to the LLM provider."""
        ...


class Location(BaseModel):
    """Location information for JSON Resume format."""

    address: Optional[str] = None
    postalCode: Optional[str] = None
    city: Optional[str] = None
    countryCode: Optional[str] = None
    region: Optional[str] = None


class Profile(BaseModel):
    """Social profile information for JSON Resume format."""

    network: Optional[str] = None
    username: Optional[str] = None
    url: str


class Basics(BaseModel):
    """Basic information for JSON Resume format."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[Location] = None
    profiles: Optional[List[Profile]] = None


class Work(BaseModel):
    """Work experience for JSON Resume format."""

    name: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Volunteer(BaseModel):
    """Volunteer experience for JSON Resume format."""

    organization: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Education(BaseModel):
    """Education information for JSON Resume format."""

    institution: Optional[str] = None
    url: Optional[str] = None
    area: Optional[str] = None
    studyType: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    score: Optional[str] = None
    courses: Optional[List[str]] = None


class Award(BaseModel):
    """Award information for JSON Resume format."""

    title: Optional[str] = None
    date: Optional[str] = None
    awarder: Optional[str] = None
    summary: Optional[str] = None


class Certificate(BaseModel):
    """Certificate information for JSON Resume format."""

    name: Optional[str] = None
    date: Optional[str] = None
    issuer: Optional[str] = None
    url: Optional[str] = None


class Publication(BaseModel):
    """Publication information for JSON Resume format."""

    name: Optional[str] = None
    publisher: Optional[str] = None
    releaseDate: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class Skill(BaseModel):
    """Skill information for JSON Resume format."""

    name: Optional[str] = None
    level: Optional[str] = None
    keywords: Optional[List[str]] = None


class Language(BaseModel):
    """Language information for JSON Resume format."""

    language: Optional[str] = None
    fluency: Optional[str] = None


class Interest(BaseModel):
    """Interest information for JSON Resume format."""

    name: Optional[str] = None
    keywords: Optional[List[str]] = None


class Reference(BaseModel):
    """Reference information for JSON Resume format."""

    name: Optional[str] = None
    reference: Optional[str] = None


class Project(BaseModel):
    """Project information for JSON Resume format."""

    name: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    highlights: Optional[List[str]] = None
    url: Optional[str] = None
    technologies: Optional[List[str]] = None
    skills: Optional[List[str]] = None


class BasicsSection(BaseModel):
    """Basics section containing basic information."""

    basics: Optional[Basics] = None


class WorkSection(BaseModel):
    """Work section containing a list of work experiences."""

    work: Optional[List[Work]] = None


class EducationSection(BaseModel):
    """Education section containing a list of education entries."""

    education: Optional[List[Education]] = None


class SkillsSection(BaseModel):
    """Skills section containing a list of skill categories."""

    skills: Optional[List[Skill]] = None


class ProjectsSection(BaseModel):
    """Projects section containing a list of projects."""

    projects: Optional[List[Project]] = None


class AwardsSection(BaseModel):
    """Awards section containing a list of awards."""

    awards: Optional[List[Award]] = None


class JSONResume(BaseModel):
    """Complete JSON Resume format model."""

    basics: Optional[Basics] = None
    work: Optional[List[Work]] = None
    volunteer: Optional[List[Volunteer]] = None
    education: Optional[List[Education]] = None
    awards: Optional[List[Award]] = None
    certificates: Optional[List[Certificate]] = None
    publications: Optional[List[Publication]] = None
    skills: Optional[List[Skill]] = None
    languages: Optional[List[Language]] = None
    interests: Optional[List[Interest]] = None
    references: Optional[List[Reference]] = None
    projects: Optional[List[Project]] = None


CATEGORY_MAX = 25


class CategoryScore(BaseModel):
    """One rubric category. Scores are clamped rather than rejected.

    A model that hallucinates `{"score": 999, "max": 25}` used to validate and
    rank first; clamping keeps that row usable instead of discarding the whole
    applicant over one bad number.
    """

    score: float = Field(ge=0, description="Score achieved in this category")
    max: int = Field(gt=0, description="Maximum possible score")
    evidence: str = Field(min_length=1, description="Evidence supporting the score")

    @field_validator("max")
    @classmethod
    def _normalize_max(cls, v: int) -> int:
        return CATEGORY_MAX

    @model_validator(mode="after")
    def _clamp_score(self):
        if self.score > self.max:
            self.score = float(self.max)
        return self


class Scores(BaseModel):
    motivation_fit: CategoryScore
    collaboration_perspective: CategoryScore
    values_judgment: CategoryScore
    commitments_experience: CategoryScore


class EvaluationData(BaseModel):
    """Exactly what the model is asked to produce -- nothing more.

    GPA and resume metadata are deliberately absent: they are decided by app
    code, and including them here only forced the model to emit dead tokens
    that were immediately overwritten.
    """

    scores: Scores
    key_strengths: List[str] = Field(min_length=1, max_length=5)
    areas_for_improvement: List[str] = Field(min_length=1, max_length=3)


class OpenAICompatibleProvider:
    """Generic OpenAI-chat-compatible LLM provider.

    Works for Ollama (/v1), Gemini (/v1beta/openai), OpenAI, Groq, OpenRouter,
    DeepSeek, LM Studio, vLLM, etc. via a configurable base_url. Adapts the
    response to the {"message": {"content": ...}} shape the evaluator expects.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        structured_output: str = "json_schema",
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.structured_output = structured_output
        self.extra_body = extra_body or {}

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import requests
        import time
        import random

        import logging

        log = logging.getLogger(__name__)

        options = options or {}

        # extra_body is applied first so provider config can supply defaults but
        # can never clobber the fields this method sets deliberately below.
        body: Dict[str, Any] = dict(self.extra_body)
        body.update({"model": model, "messages": messages, "stream": False})
        if "temperature" in options:
            body["temperature"] = options["temperature"]
        if "top_p" in options:
            body["top_p"] = options["top_p"]

        # Structured-output translation: evaluator passes format=<json schema>.
        if "format" in kwargs and self.structured_output != "none":
            schema = kwargs["format"]
            if self.structured_output == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema},
                }
            elif self.structured_output == "json_object":
                body["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        MAX_RETRIES = 5
        BASE_DELAY = 2.0  # seconds — base for exponential backoff
        MAX_DELAY = 60.0  # cap so we never wait more than a minute

        def _backoff(attempt: int, retry_after: Optional[str] = None) -> float:
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            if retry_after:
                # RFC 7231 allows an HTTP-date here, which is not a float.
                try:
                    delay = min(float(retry_after), MAX_DELAY)
                except (TypeError, ValueError):
                    pass
            return round(delay * random.uniform(0.8, 1.2), 2)

        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            is_last = attempt == MAX_RETRIES - 1
            try:
                # (connect, read): fail fast when the server is down, stay
                # patient while a local model is generating.
                response = requests.post(
                    url, json=body, headers=headers, timeout=(10, 300)
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                # The dominant failure mode for a local Ollama backend: the
                # server restarts or a long generation stalls. Previously fatal
                # on the first occurrence, which killed the whole job.
                last_error = e
                if is_last:
                    break
                wait = _backoff(attempt)
                log.warning(
                    "LLM request failed (%s), retrying in %.1fs [%s/%s]",
                    type(e).__name__, wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
                continue

            # 429 (rate limit) and 5xx (server hiccup) are both transient.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = requests.HTTPError(
                    f"HTTP {response.status_code} from {url}", response=response
                )
                if is_last:
                    break
                wait = _backoff(attempt, response.headers.get("Retry-After"))
                log.warning(
                    "LLM returned HTTP %s, retrying in %.1fs [%s/%s]",
                    response.status_code, wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as e:
                raise ValueError(
                    f"Non-JSON response from {url}: {response.text[:300]!r}"
                ) from e

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise ValueError(f"Unexpected response shape from {url}: {data}")
            return {"message": {"role": "assistant", "content": content}}

        raise RuntimeError(
            f"LLM request to {url} failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error
