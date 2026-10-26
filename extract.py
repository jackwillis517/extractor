import time
from typing import Any

import instructor
from instructor.core import IncompleteOutputException, InstructorRetryException
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    NotFoundError,
    OpenAI,
)

from schemas import Invoice, JobSearchResults

SCHEMAS = {"job_posting": JobSearchResults, "invoice": Invoice}


class ExtractionError(Exception):
    """Extraction failed. `stats` holds whatever was measured before the failure."""

    def __init__(self, message: str, stats: dict[str, Any] | None = None):
        super().__init__(message)
        self.stats = stats or {}


def extract(
    text: str, prompt: str, model: str, schema: str
) -> tuple[JobSearchResults | Invoice, Any]:
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema {schema!r}. Choose from {list(SCHEMAS)}")
    if not text.strip():
        raise ValueError("Input text is empty")

    start = time.perf_counter()
    retry_stats = {"attempts": 0, "validation_errors": []}

    def on_attempt(*args, **kwargs):
        retry_stats["attempts"] += 1

    def on_parse_error(error):
        retry_stats["validation_errors"].append(str(error)[:200])

    def build_stats(usage=None) -> dict[str, Any]:
        latency = time.perf_counter() - start
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return {
            "latency_s": round(latency, 2),
            "tokens_in": getattr(usage, "prompt_tokens", 0) or 0,
            "tokens_out": tokens_out,
            "tok_per_s": round(tokens_out / latency, 1) if latency > 0 else 0.0,
            "attempts": retry_stats["attempts"],
            "retried": retry_stats["attempts"] > 1,
            "errors": retry_stats["validation_errors"],
        }

    ollama_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=300,
        max_retries=0,
    )

    client = instructor.from_openai(
        ollama_client, mode=instructor.Mode.JSON_SCHEMA, max_tokens=32768
    )

    client.on("completion:kwargs", on_attempt)  # fires once per call to the model
    client.on("parse:error", on_parse_error)

    try:
        result, completion = client.chat.completions.create_with_completion(
            model=model,
            response_model=JobSearchResults if schema == "job_posting" else Invoice,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{text}",
                }
            ],
            max_retries=3,
        )
    except IncompleteOutputException as e:
        raise ExtractionError(
            "Output was cut off at max_tokens. Raise max_tokens/num_ctx or split the input.",
            build_stats(),
        ) from e

    except InstructorRetryException as e:
        last = (
            retry_stats["validation_errors"][-1]
            if retry_stats["validation_errors"]
            else "unknown"
        )
        raise ExtractionError(
            f"Output failed validation after {e.n_attempts} attempts. Last error: {last}",
            build_stats(e.total_usage),
        ) from e
    except (
        APITimeoutError
    ) as e:  # must come before APIConnectionError (it's a subclass)
        raise ExtractionError("Ollama timed out", build_stats()) from e
    except APIConnectionError as e:
        raise ExtractionError(
            "Can't reach Ollama at localhost:11434. Is `ollama serve` running?",
            build_stats(),
        ) from e
    except NotFoundError as e:  # must come before APIStatusError (it's a subclass)
        raise ExtractionError(
            f"Model {model!r} not found. Run `ollama pull {model}`", build_stats()
        ) from e
    except APIStatusError as e:
        raise ExtractionError(
            f"Ollama returned {e.status_code}: {e.message}", build_stats()
        ) from e

    return result, build_stats(completion.usage)
