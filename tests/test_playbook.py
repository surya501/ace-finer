"""Tests for playbook/schema.py and playbook/store.py."""

import pytest
import tempfile
import shutil
from pathlib import Path
from playbook.schema import Rule
from playbook.store import PlaybookStore


class TestRule:
    """Tests for Rule dataclass."""

    def test_create_generates_uuid(self):
        """Rule.create() generates a UUID."""
        rule = Rule.create(
            content="When you see 'revenue', tag it as B-Revenue",
            trigger_context="revenue context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        assert len(rule.rule_id) == 36  # UUID format
        assert "-" in rule.rule_id

    def test_to_metadata(self):
        """to_metadata() serializes correctly."""
        rule = Rule(
            rule_id="test-id",
            content="Test rule",
            trigger_context="test context",
            target_entities=["Revenue", "Assets"],
            error_type="classification",
            success_count=5,
            failure_count=2,
            created_at="2024-01-01T00:00:00"
        )
        meta = rule.to_metadata()

        assert meta["trigger_context"] == "test context"
        assert meta["target_entities"] == "Revenue,Assets"
        assert meta["error_type"] == "classification"
        assert meta["success_count"] == 5
        assert meta["failure_count"] == 2

    def test_from_query_result(self):
        """from_query_result() deserializes correctly."""
        metadata = {
            "trigger_context": "test context",
            "target_entities": "Revenue,Assets",
            "error_type": "boundary",
            "success_count": 3,
            "failure_count": 1,
            "created_at": "2024-01-01T00:00:00",
            "last_used": "2024-01-02T00:00:00"
        }
        rule = Rule.from_query_result(
            rule_id="test-id",
            content="Test rule",
            metadata=metadata
        )

        assert rule.rule_id == "test-id"
        assert rule.content == "Test rule"
        assert rule.target_entities == ["Revenue", "Assets"]
        assert rule.success_count == 3
        assert rule.last_used == "2024-01-02T00:00:00"

    def test_utility_score_laplace(self):
        """utility_score() uses Laplace smoothing."""
        # (success + 1) / (success + failure + 2)
        rule = Rule(
            rule_id="1", content="", trigger_context="",
            target_entities=[], success_count=0, failure_count=0
        )
        assert rule.utility_score() == 0.5  # (0+1)/(0+0+2) = 0.5

        rule.success_count = 8
        rule.failure_count = 2
        assert rule.utility_score() == pytest.approx(0.75)  # (8+1)/(8+2+2) = 0.75

    def test_roundtrip_serialization(self):
        """Rule survives to_metadata -> from_query_result."""
        original = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["A", "B"],
            error_type="hallucination"
        )
        original.success_count = 10
        original.failure_count = 5

        meta = original.to_metadata()
        restored = Rule.from_query_result(original.rule_id, original.content, meta)

        assert restored.content == original.content
        assert restored.target_entities == original.target_entities
        assert restored.success_count == original.success_count


class TestPlaybookStore:
    """Tests for PlaybookStore ChromaDB wrapper."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        temp_dir = tempfile.mkdtemp()
        store = PlaybookStore(path=temp_dir)
        yield store
        shutil.rmtree(temp_dir)

    def test_add_and_count(self, temp_store):
        """add_rule() increments count."""
        assert temp_store.count() == 0

        rule = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        temp_store.add_rule(rule)

        assert temp_store.count() == 1

    def test_add_duplicate_ignored(self, temp_store):
        """Adding same rule_id twice doesn't duplicate."""
        rule = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        temp_store.add_rule(rule)
        temp_store.add_rule(rule)

        assert temp_store.count() == 1

    def test_retrieve_by_similarity(self, temp_store):
        """retrieve() returns semantically similar rules."""
        rule1 = Rule.create(
            content="Revenue recognition for financial statements",
            trigger_context="financial",
            target_entities=["Revenue"],
            error_type="omission"
        )
        rule2 = Rule.create(
            content="Asset depreciation calculation",
            trigger_context="assets",
            target_entities=["Assets"],
            error_type="classification"
        )
        temp_store.add_rule(rule1)
        temp_store.add_rule(rule2)

        # Query about revenue should return revenue rule first
        results = temp_store.retrieve("company revenue earnings", top_k=2)
        assert len(results) == 2
        assert any("Revenue" in r.content for r in results)

    def test_retrieve_empty_store(self, temp_store):
        """retrieve() returns empty list on empty store."""
        results = temp_store.retrieve("anything", top_k=5)
        assert results == []

    def test_update_stats_success(self, temp_store):
        """update_stats() increments success count."""
        rule = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        temp_store.add_rule(rule)

        temp_store.update_stats(rule.rule_id, success=True)
        temp_store.update_stats(rule.rule_id, success=True)

        retrieved = temp_store.retrieve("Test rule", top_k=1)[0]
        assert retrieved.success_count == 2

    def test_update_stats_failure(self, temp_store):
        """update_stats() increments failure count."""
        rule = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        temp_store.add_rule(rule)

        temp_store.update_stats(rule.rule_id, success=False)

        retrieved = temp_store.retrieve("Test rule", top_k=1)[0]
        assert retrieved.failure_count == 1

    def test_delete_rule(self, temp_store):
        """delete_rule() removes rule."""
        rule = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        temp_store.add_rule(rule)
        assert temp_store.count() == 1

        temp_store.delete_rule(rule.rule_id)
        assert temp_store.count() == 0

    def test_get_all_rules(self, temp_store):
        """get_all_rules() returns all rules."""
        for i in range(3):
            rule = Rule.create(
                content=f"Rule {i}",
                trigger_context="context",
                target_entities=["Revenue"],
                error_type="omission"
            )
            temp_store.add_rule(rule)

        all_rules = temp_store.get_all_rules()
        assert len(all_rules) == 3

    def test_get_embedding(self, temp_store):
        """get_embedding() returns a vector."""
        embedding = temp_store.get_embedding("test text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_checkpoint(self, temp_store):
        """checkpoint() creates a copy of the database."""
        rule = Rule.create(
            content="Test rule",
            trigger_context="context",
            target_entities=["Revenue"],
            error_type="omission"
        )
        temp_store.add_rule(rule)

        checkpoint_dir = tempfile.mkdtemp()
        checkpoint_path = f"{checkpoint_dir}/test_checkpoint"

        try:
            temp_store.checkpoint(checkpoint_path)
            assert Path(checkpoint_path).exists()
        finally:
            shutil.rmtree(checkpoint_dir)
