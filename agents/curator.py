"""Curator agent for playbook maintenance."""

import logging
from dataclasses import dataclass
from agents.llm import LLMClient
from playbook.store import PlaybookStore

log = logging.getLogger(__name__)


@dataclass
class CurationReport:
    """Summary of curation actions."""
    total_rules: int
    pruned_low_utility: int
    pruned_duplicates: int

    def __str__(self) -> str:
        return (
            f"rules={self.total_rules}, "
            f"pruned_low_utility={self.pruned_low_utility}, "
            f"pruned_duplicates={self.pruned_duplicates}"
        )


class Curator:
    """
    Maintains playbook quality by pruning low-utility and duplicate rules.

    The Curator:
    1. Identifies rules with low utility scores
    2. Detects semantic duplicates
    3. Removes underperforming rules
    """

    def __init__(
        self,
        store: PlaybookStore,
        llm: LLMClient,
        min_utility: float = 0.3,
        duplicate_threshold: float = 0.9
    ):
        """
        Initialize the curator.

        Args:
            store: PlaybookStore to curate
            llm: LLM client (for potential merging, not currently used)
            min_utility: Minimum utility score to keep a rule
            duplicate_threshold: Cosine similarity threshold for duplicates
        """
        self.store = store
        self.llm = llm
        self.min_utility = min_utility
        self.duplicate_threshold = duplicate_threshold

    async def run(self) -> CurationReport:
        """
        Run curation on the playbook.

        Returns:
            CurationReport with actions taken
        """
        all_rules = self.store.get_all_rules()
        total_rules = len(all_rules)

        pruned_low_utility = 0
        pruned_duplicates = 0

        # Prune low utility rules (need minimum interactions first)
        for rule in all_rules:
            total_interactions = rule.success_count + rule.failure_count
            if total_interactions >= 5 and rule.utility_score() < self.min_utility:
                self.store.delete_rule(rule.rule_id)
                pruned_low_utility += 1
                log.info(f"Pruned low-utility rule: {rule.rule_id[:8]}... (score={rule.utility_score():.2f})")

        # Reload rules after pruning
        all_rules = self.store.get_all_rules()

        # Detect and remove duplicates
        seen_embeddings = []
        seen_ids = []

        for rule in all_rules:
            embedding = self.store.get_embedding(rule.content)

            is_duplicate = False
            for i, seen_emb in enumerate(seen_embeddings):
                sim = self._cosine_similarity(embedding, seen_emb)
                if sim > self.duplicate_threshold:
                    # Keep the one with higher utility
                    existing_rule = next(
                        (r for r in all_rules if r.rule_id == seen_ids[i]), None
                    )
                    if existing_rule and rule.utility_score() <= existing_rule.utility_score():
                        self.store.delete_rule(rule.rule_id)
                        pruned_duplicates += 1
                        log.info(f"Pruned duplicate rule: {rule.rule_id[:8]}...")
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_embeddings.append(embedding)
                seen_ids.append(rule.rule_id)

        final_count = self.store.count()

        return CurationReport(
            total_rules=final_count,
            pruned_low_utility=pruned_low_utility,
            pruned_duplicates=pruned_duplicates
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
