"""Circuit breakers and safety guards for ACE framework."""

from collections import deque
from dataclasses import dataclass, field
import logging
import numpy as np

log = logging.getLogger(__name__)


class GuardError(Exception):
    """Raised when a guard condition is violated."""
    pass


@dataclass
class Guards:
    """
    Safety guards for the ACE framework.

    Monitors:
    - Cost budget (total API spend)
    - Rule explosion (ratio of rules created to samples)
    - Parse failures (consecutive LLM parse failures)
    - Duplicate rules (via embedding similarity)
    """
    max_cost: float = 5.0
    max_rule_ratio: float = 0.5
    max_parse_failures: int = 5
    warmup: int = 100
    duplicate_threshold: float = 0.95
    max_recent_embeddings: int = 10

    # Counters (initialized in __post_init__)
    samples: int = field(default=0, init=False)
    rules_created: int = field(default=0, init=False)
    consecutive_parse_failures: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)
    recent_embeddings: deque = field(default=None, init=False)

    def __post_init__(self):
        self.recent_embeddings = deque(maxlen=self.max_recent_embeddings)

    def record_sample(self):
        """Record that a sample was processed."""
        self.samples += 1

    def record_rule_created(self):
        """Record that a new rule was created."""
        self.rules_created += 1

    def record_parse_failure(self):
        """Record a parse failure."""
        self.consecutive_parse_failures += 1

    def record_parse_success(self):
        """Record a successful parse, resetting failure counter."""
        self.consecutive_parse_failures = 0

    def record_cost(self, cost: float):
        """Record API cost. Called by LLM client after each request."""
        self.total_cost += cost

    def check_duplicate(self, embedding: list[float]) -> bool:
        """
        Check if embedding is too similar to recent ones.

        Args:
            embedding: Vector representation of the rule content

        Returns:
            True if OK to add (not a duplicate), False if duplicate
        """
        emb = np.array(embedding)
        emb_norm = np.linalg.norm(emb)

        if emb_norm == 0:
            return True  # Zero vector, allow it

        for recent in self.recent_embeddings:
            recent_norm = np.linalg.norm(recent)
            if recent_norm == 0:
                continue
            sim = np.dot(emb, recent) / (emb_norm * recent_norm)
            if sim > self.duplicate_threshold:
                log.warning(f"Duplicate rule detected (sim={sim:.3f}), skipping")
                return False

        self.recent_embeddings.append(emb)
        return True

    def check_all(self, step: int):
        """
        Check all guards, raise GuardError if any violated.

        Args:
            step: Current step number for logging
        """
        # Cost guard
        if self.total_cost > self.max_cost:
            raise GuardError(f"Budget exceeded: ${self.total_cost:.2f} > ${self.max_cost}")

        # Parse failure guard
        if self.consecutive_parse_failures >= self.max_parse_failures:
            raise GuardError(f"{self.max_parse_failures} consecutive parse failures")

        # Rule ratio guard (only after warmup)
        if self.samples >= self.warmup:
            ratio = self.rules_created / self.samples
            if ratio > self.max_rule_ratio:
                raise GuardError(f"Rule ratio {ratio:.2f} > {self.max_rule_ratio}")
