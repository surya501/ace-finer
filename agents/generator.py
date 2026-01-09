"""Generator agent for predicting IOB2 labels."""

import json
import logging
from agents.llm import LLMClient
from playbook.schema import Rule

log = logging.getLogger(__name__)


class Generator:
    """Predicts IOB2 labels for tokens using LLM with retrieved rules."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def predict(self, tokens: list[str], rules: list[Rule]) -> tuple[list[str], bool]:
        """
        Predict IOB2 labels for tokens.

        Args:
            tokens: List of tokens to tag
            rules: Retrieved rules to include in prompt

        Returns:
            (labels, parse_failed): IOB2 labels and whether parsing failed
        """
        prompt = self._build_prompt(tokens, rules)
        response = await self.llm.complete([{"role": "user", "content": prompt}])

        labels, failed = self._parse(response, len(tokens))
        if failed:
            log.warning(f"Parse failed: {response[:100]}...")

        return labels, failed

    def _build_prompt(self, tokens: list[str], rules: list[Rule]) -> str:
        """Build prompt with XML-tagged tokens and rules."""
        xml_tokens = " ".join(f'<t id="{i}">{t}</t>' for i, t in enumerate(tokens))
        rules_text = "\n".join(f"- {r.content}" for r in rules) if rules else "(none)"

        return f"""Tag tokens with XBRL entity labels (IOB2 format).

Rules:
{rules_text}

Tokens:
{xml_tokens}

Output JSON mapping token IDs to non-O labels, e.g. {{"3": "B-Revenue"}}
Return {{}} if all tokens are O."""

    def _parse(self, response: str, n: int) -> tuple[list[str], bool]:
        """
        Parse JSON response, return (labels, parse_failed).

        Args:
            response: LLM response text
            n: Number of tokens (length of labels list)

        Returns:
            (labels, failed): List of IOB2 labels and whether parsing failed
        """
        labels = ["O"] * n

        try:
            # Find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                for idx_str, label in data.items():
                    try:
                        idx = int(idx_str)
                        if 0 <= idx < n:
                            labels[idx] = str(label)
                    except (ValueError, TypeError):
                        continue

                return labels, False

        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"JSON parse error: {e}")

        return labels, True
