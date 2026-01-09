"""Rule dataclass for playbook storage."""

from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Rule:
    """
    A rule in the playbook that guides entity extraction.

    Rules are learned from errors and stored in ChromaDB for retrieval.
    """
    rule_id: str
    content: str  # The actual rule text
    trigger_context: str  # Context that triggers this rule
    target_entities: list[str]  # Entity types this rule applies to
    error_type: str = "classification"  # Type of error this fixes
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used: str | None = None

    @staticmethod
    def create(
        content: str,
        trigger_context: str,
        target_entities: list[str],
        error_type: str
    ) -> "Rule":
        """Factory method to create a new rule with generated ID."""
        return Rule(
            rule_id=str(uuid.uuid4()),
            content=content,
            trigger_context=trigger_context,
            target_entities=target_entities,
            error_type=error_type
        )

    def to_metadata(self) -> dict:
        """Serialize to ChromaDB metadata format."""
        return {
            "trigger_context": self.trigger_context,
            "target_entities": ",".join(self.target_entities),
            "error_type": self.error_type,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "last_used": self.last_used or ""
        }

    @classmethod
    def from_query_result(cls, rule_id: str, content: str, metadata: dict) -> "Rule":
        """Deserialize from ChromaDB query result."""
        target_entities = metadata.get("target_entities", "")
        return cls(
            rule_id=rule_id,
            content=content,
            trigger_context=metadata.get("trigger_context", ""),
            target_entities=target_entities.split(",") if target_entities else [],
            error_type=metadata.get("error_type", "classification"),
            success_count=metadata.get("success_count", 0),
            failure_count=metadata.get("failure_count", 0),
            created_at=metadata.get("created_at", ""),
            last_used=metadata.get("last_used") or None
        )

    def utility_score(self) -> float:
        """
        Calculate utility score using Laplace smoothing.

        Returns a value between 0 and 1 representing the rule's usefulness.
        Higher is better.
        """
        return (self.success_count + 1) / (self.success_count + self.failure_count + 2)
