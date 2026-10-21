"""A small LLM client for any OpenAI-compatible chat endpoint (Groq, Gemini, OpenRouter, Ollama...).

Configuration comes from the environment or a `.env` file in the working directory:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, and optionally LLM_MIN_INTERVAL (seconds between calls).

Every response is cached on disk, so re-running an experiment makes no new API calls.
The API key is never logged.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        cache_dir: Path | None = Path(".cache/llm"),
        min_interval: float = 2.5,
        max_retries: int = 6,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_call = 0.0
        self.calls_made = 0
        self.cache_hits = 0
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, **kwargs) -> "LLMClient":
        load_dotenv()
        missing = [v for v in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL") if not os.environ.get(v)]
        if missing:
            raise LLMError(
                f"missing {', '.join(missing)}. Put them in a .env file in the project root, e.g.\n"
                "LLM_API_KEY=...\nLLM_BASE_URL=https://api.groq.com/openai/v1\nLLM_MODEL=llama-3.3-70b-versatile"
            )
        return cls(
            os.environ["LLM_API_KEY"],
            os.environ["LLM_BASE_URL"],
            os.environ["LLM_MODEL"],
            min_interval=float(os.environ.get("LLM_MIN_INTERVAL", 2.5)),
            **kwargs,
        )

    def _cache_path(self, payload: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cache_path = self._cache_path(payload)
        if cache_path is not None and cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:
                logger.warning("LLM request failed (%s), retrying", type(exc).__name__)
                time.sleep(2**attempt)
                continue

            if response.status_code == 200:
                text = (response.json()["choices"][0]["message"].get("content") or "").strip()
                self.calls_made += 1
                if not text:
                    # Reasoning models can spend the whole token budget thinking and return nothing.
                    # Don't cache that, or every rerun would replay the empty answer.
                    logger.warning("LLM returned an empty reply (model %s); not caching it", self.model)
                    return text
                if cache_path is not None:
                    cache_path.write_text(json.dumps({"text": text}), encoding="utf-8")
                return text
            if response.status_code in (401, 403):
                raise LLMError(f"authentication failed (HTTP {response.status_code}); check LLM_API_KEY")
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 2 ** (attempt + 1)
                logger.warning("LLM HTTP %d, waiting %.0fs (attempt %d)", response.status_code, delay, attempt + 1)
                time.sleep(min(delay, 60))
                continue
            raise LLMError(f"LLM request failed: HTTP {response.status_code}: {response.text[:200]}")

        raise LLMError(f"LLM request failed after {self.max_retries} attempts")
