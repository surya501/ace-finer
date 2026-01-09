"""LLM client for OpenRouter API with retry and cost tracking."""

import asyncio
import logging
import time
from typing import Callable
import openai

log = logging.getLogger(__name__)

# Model pricing per 1M tokens (input, output)
MODEL_PRICING = {
    # Free models
    "nvidia/nemotron-nano-9b-v2:free": (0.0, 0.0),
    # Paid models (for reference)
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "meta-llama/llama-3.3-70b-instruct": (0.30, 0.30),
}

DEFAULT_MODEL = "openai/gpt-oss-20b"

# OpenRouter rate limits (paid models with credits)
DEFAULT_RPM = 200


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute: int = DEFAULT_RPM):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make another request."""
        async with self._lock:
            now = time.monotonic()
            wait_time = self.last_request + self.min_interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request = time.monotonic()


class LLMClient:
    """
    Async LLM client for OpenRouter with OpenAI-compatible API.

    Features:
    - Rate limiting for free tier (20 req/min)
    - Exponential backoff on rate limits
    - Cost tracking via callback
    - Configurable model and provider
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        on_cost: Callable[[float], None] | None = None,
        requests_per_minute: int = DEFAULT_RPM
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key (OpenRouter or other provider)
            model: Model name to use
            base_url: API base URL (OpenRouter, Groq, etc.)
            on_cost: Callback invoked with cost after each request
            requests_per_minute: Rate limit (default: 20 for free tier)
        """
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.on_cost = on_cost
        self.rate_limiter = RateLimiter(requests_per_minute)

    async def complete(self, messages: list[dict], retries: int = 5) -> str:
        """
        Make completion with rate limiting and exponential backoff.

        Args:
            messages: Chat messages in OpenAI format
            retries: Number of retry attempts

        Returns:
            Response content string

        Raises:
            RuntimeError: If all retries fail
        """
        # Wait for rate limiter
        await self.rate_limiter.acquire()

        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )

                # Calculate and report cost
                cost = self._calc_cost(resp.usage)
                if self.on_cost:
                    self.on_cost(cost)

                if resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content
                log.warning("Empty response from API, retrying...")
                await asyncio.sleep(1)
                continue

            except openai.RateLimitError:
                wait = 2 ** (attempt + 2)  # Start at 4s, then 8, 16, 32, 64
                log.warning(f"Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)

            except openai.APIError as e:
                log.warning(f"API error: {e}, retry {attempt + 1}/{retries}")
                await asyncio.sleep(1)

        raise RuntimeError("LLM call failed after retries")

    def _calc_cost(self, usage) -> float:
        """
        Calculate cost based on model pricing.

        Returns 0 for free models, otherwise uses MODEL_PRICING table.
        """
        if usage is None:
            return 0.0

        pricing = MODEL_PRICING.get(self.model, (0.0, 0.0))
        input_cost = (usage.prompt_tokens * pricing[0]) / 1_000_000
        output_cost = (usage.completion_tokens * pricing[1]) / 1_000_000
        return input_cost + output_cost
