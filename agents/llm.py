"""LLM client for Groq API with retry and cost tracking."""

import asyncio
import logging
from typing import Callable
import openai

log = logging.getLogger(__name__)


class LLMClient:
    """
    Async LLM client for Groq with OpenAI-compatible API.

    Features:
    - Exponential backoff on rate limits
    - Cost tracking via callback
    - Configurable model
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        on_cost: Callable[[float], None] | None = None
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Groq API key
            model: Model name to use
            on_cost: Callback invoked with cost after each request
        """
        self.client = openai.AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.model = model
        self.on_cost = on_cost

    async def complete(self, messages: list[dict], retries: int = 3) -> str:
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
                wait = 2 ** attempt
                log.warning(f"Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)

            except openai.APIError as e:
                log.warning(f"API error: {e}, retry {attempt + 1}/{retries}")
                await asyncio.sleep(1)

        raise RuntimeError("LLM call failed after retries")

    def _calc_cost(self, usage) -> float:
        """
        Calculate cost based on Groq pricing.

        Groq pricing for llama-3.3-70b-versatile ($/1M tokens):
        - Input: $0.59
        - Output: $0.79
        """
        return (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000
