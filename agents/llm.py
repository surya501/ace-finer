"""LLM client for OpenRouter API with retry and cost tracking."""

import asyncio
import logging
import os
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

DEFAULT_MODEL = "nvidia/nemotron-nano-9b-v2:free"


class LLMClient:
    """
    Async LLM client for OpenRouter with OpenAI-compatible API.

    Features:
    - Exponential backoff on rate limits
    - Cost tracking via callback
    - Configurable model and provider
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        on_cost: Callable[[float], None] | None = None
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key (OpenRouter or other provider)
            model: Model name to use
            base_url: API base URL (OpenRouter, Groq, etc.)
            on_cost: Callback invoked with cost after each request
        """
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.on_cost = on_cost

    async def complete(self, messages: list[dict], retries: int = 5) -> str:
        """
        Make completion with exponential backoff.

        Args:
            messages: Chat messages in OpenAI format
            retries: Number of retry attempts

        Returns:
            Response content string

        Raises:
            RuntimeError: If all retries fail
        """
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

                return resp.choices[0].message.content

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
