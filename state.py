"""State dataclass for ACE pipeline."""

from dataclasses import dataclass, field
from playbook.schema import Rule


@dataclass
class State:
    """
    State passed through each step of processing.

    Captures input, processing results, evaluation, and reflection.
    """
    # Input
    sample_id: int
    sentence: str
    tokens: list[str]
    ground_truth: list[str]

    # Processing
    retrieved_rules: list[Rule] = field(default_factory=list)
    predictions: list[str] | None = None

    # Evaluation
    is_correct: bool | None = None
    error_type: str | None = None  # boundary/classification/omission/hallucination

    # Reflection
    new_rule: Rule | None = None

    # Metrics (for logging)
    parse_failed: bool = False
