"""Reflector agent for generating rules from errors."""

import json
import logging
from agents.llm import LLMClient
from playbook.schema import Rule
from state import State

log = logging.getLogger(__name__)


class Reflector:
    """
    Analyzes prediction errors and generates rules to prevent them.

    The Reflector:
    1. Analyzes what went wrong in a prediction
    2. Generates a rule to fix similar errors
    3. Validates the rule actually helps
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate_rule(self, state: State) -> Rule | None:
        """
        Generate a rule from an error.

        Args:
            state: State with error (is_correct=False)

        Returns:
            Rule object or None if generation failed
        """
        if state.is_correct:
            return None

        prompt = self._build_prompt(state)
        response = await self.llm.complete([{"role": "user", "content": prompt}])

        return self._parse_rule(response, state.error_type or "classification")

    async def validate(self, rule: Rule, state: State, generator) -> bool:
        """
        Validate that a rule fixes the error.

        Args:
            rule: Rule to validate
            state: Original state with error
            generator: Generator to re-run prediction

        Returns:
            True if rule fixes the error
        """
        # Re-run prediction with new rule added
        test_rules = state.retrieved_rules + [rule]
        predictions, parse_failed = await generator.predict(state.tokens, test_rules)

        if parse_failed:
            return False

        # Check if prediction improved
        return predictions == state.ground_truth

    def _build_prompt(self, state: State) -> str:
        """Build prompt for rule generation."""
        xml_tokens = " ".join(f'<t id="{i}">{t}</t>' for i, t in enumerate(state.tokens))

        pred_str = ", ".join(
            f'{i}: {l}' for i, l in enumerate(state.predictions or []) if l != "O"
        ) or "(none)"
        truth_str = ", ".join(
            f'{i}: {l}' for i, l in enumerate(state.ground_truth) if l != "O"
        ) or "(none)"

        rules_str = "\n".join(f"- {r.content}" for r in state.retrieved_rules) or "(none)"

        return f"""Analyze this XBRL tagging error and create a rule to prevent it.

Tokens:
{xml_tokens}

Predicted: {pred_str}
Expected: {truth_str}

Error type: {state.error_type}

Current rules:
{rules_str}

Create a rule to help tag similar sentences correctly.
Output JSON with:
- "content": The rule text (1-2 sentences, actionable)
- "trigger_context": Keywords/patterns that trigger this rule
- "target_entities": List of entity types this rule helps with

Example output:
{{"content": "When you see 'net revenue' or 'total revenue', tag the full phrase as B-Revenue I-Revenue", "trigger_context": "revenue, net revenue, total revenue", "target_entities": ["Revenue"]}}"""

    def _parse_rule(self, response: str, error_type: str) -> Rule | None:
        """Parse rule from LLM response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1

            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                content = data.get("content", "")
                trigger_context = data.get("trigger_context", "")
                target_entities = data.get("target_entities", [])

                if not content:
                    return None

                if isinstance(target_entities, str):
                    target_entities = [target_entities]

                return Rule.create(
                    content=content,
                    trigger_context=trigger_context,
                    target_entities=target_entities,
                    error_type=error_type
                )

        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"Rule parse error: {e}")

        return None
