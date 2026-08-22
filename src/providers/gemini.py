import logging
import random
import time

from google import genai
from google.genai import types

from src.config import require_env

log = logging.getLogger(__name__)

MAX_RETRIES = 5


class GeminiProvider:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.name = model
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            self._client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
        return self._client

    def generate(self, prompt: str, max_tokens: int) -> str:
        client = self._client_or_create()
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                        # This is grounded extraction from sections already supplied
                        # in the prompt, not a reasoning task, so a thinking budget
                        # buys nothing. Left enabled, gemini-2.5-flash spends most of
                        # max_output_tokens on invisible thinking tokens and silently
                        # starves the visible answer (measured: 767/800 tokens to
                        # thinking, 29 left for text, truncated mid-citation).
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = (2 ** attempt) + random.random()
                log.warning("generate retry %d after %.1fs: %s", attempt + 1, delay, exc)
                time.sleep(delay)
