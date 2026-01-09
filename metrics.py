"""Metrics logging for ACE framework."""

from collections import deque
from dataclasses import dataclass, field
import json
from state import State


@dataclass
class MetricsLogger:
    """
    Tracks and logs metrics during training.

    Uses a sliding window for recent accuracy calculation.
    """
    window: int = 100
    history: deque = field(default=None, init=False)
    total_correct: int = field(default=0, init=False)
    total_samples: int = field(default=0, init=False)
    error_counts: dict = field(default=None, init=False)
    parse_failures: int = field(default=0, init=False)
    rules_created: int = field(default=0, init=False)

    def __post_init__(self):
        self.history = deque(maxlen=self.window)
        self.error_counts = {
            "boundary": 0,
            "classification": 0,
            "omission": 0,
            "hallucination": 0
        }

    def record(self, state: State) -> None:
        """
        Record metrics from a processed state.

        Args:
            state: Completed State object
        """
        self.history.append(state.is_correct)
        self.total_samples += 1

        if state.is_correct:
            self.total_correct += 1
        elif state.error_type:
            self.error_counts[state.error_type] = self.error_counts.get(state.error_type, 0) + 1

        if state.parse_failed:
            self.parse_failures += 1

        if state.new_rule:
            self.rules_created += 1

    def summary(self) -> str:
        """
        Generate a summary string for logging.

        Returns:
            Formatted summary string
        """
        window_acc = sum(self.history) / len(self.history) if self.history else 0
        total_acc = self.total_correct / self.total_samples if self.total_samples else 0

        return (
            f"step={self.total_samples} | "
            f"window_acc={window_acc:.1%} | "
            f"total_acc={total_acc:.1%} | "
            f"rules={self.rules_created} | "
            f"parse_fails={self.parse_failures}"
        )

    def save(self, path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            path: File path for JSON output
        """
        with open(path, "w") as f:
            json.dump({
                "total_samples": self.total_samples,
                "total_correct": self.total_correct,
                "accuracy": self.total_correct / self.total_samples if self.total_samples else 0,
                "error_counts": self.error_counts,
                "parse_failures": self.parse_failures,
                "rules_created": self.rules_created
            }, f, indent=2)
