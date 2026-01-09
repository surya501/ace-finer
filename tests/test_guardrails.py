"""Tests for guardrails.py - Circuit breakers and safety guards."""

import pytest
import numpy as np
from guardrails import Guards, GuardError


class TestCostGuard:
    """Tests for cost budget guard."""

    def test_under_budget_passes(self):
        """Cost under limit does not raise."""
        g = Guards(max_cost=5.0)
        g.record_cost(2.0)
        g.record_sample()
        g.check_all(step=1)  # Should not raise

    def test_over_budget_raises(self):
        """Cost over limit raises GuardError."""
        g = Guards(max_cost=5.0)
        g.record_cost(6.0)
        g.record_sample()
        with pytest.raises(GuardError, match="Budget exceeded"):
            g.check_all(step=1)

    def test_cumulative_cost(self):
        """Multiple costs accumulate."""
        g = Guards(max_cost=5.0)
        g.record_cost(2.0)
        g.record_cost(2.0)
        g.record_cost(2.0)  # Total: 6.0
        g.record_sample()
        with pytest.raises(GuardError):
            g.check_all(step=1)


class TestRuleRatioGuard:
    """Tests for rule explosion guard."""

    def test_within_ratio_passes(self):
        """Rule ratio under limit passes."""
        g = Guards(max_rule_ratio=0.5, warmup=10)
        for _ in range(20):
            g.record_sample()
        for _ in range(5):  # 5/20 = 0.25 < 0.5
            g.record_rule_created()
        g.check_all(step=20)  # Should not raise

    def test_over_ratio_raises(self):
        """Rule ratio over limit raises after warmup."""
        g = Guards(max_rule_ratio=0.5, warmup=10)
        for _ in range(20):
            g.record_sample()
        for _ in range(15):  # 15/20 = 0.75 > 0.5
            g.record_rule_created()
        with pytest.raises(GuardError, match="Rule ratio"):
            g.check_all(step=20)

    def test_ratio_ignored_during_warmup(self):
        """Rule ratio not checked during warmup period."""
        g = Guards(max_rule_ratio=0.5, warmup=100)
        for _ in range(50):
            g.record_sample()
            g.record_rule_created()  # 100% ratio
        g.check_all(step=50)  # Should not raise (within warmup)


class TestParseFailureGuard:
    """Tests for consecutive parse failure guard."""

    def test_success_resets_counter(self):
        """Parse success resets failure counter."""
        g = Guards(max_parse_failures=3)
        g.record_parse_failure()
        g.record_parse_failure()
        g.record_parse_success()
        g.record_parse_failure()
        g.record_sample()
        g.check_all(step=1)  # Should not raise (only 1 consecutive)

    def test_consecutive_failures_raises(self):
        """Too many consecutive failures raises."""
        g = Guards(max_parse_failures=3)
        g.record_parse_failure()
        g.record_parse_failure()
        g.record_parse_failure()
        g.record_sample()
        with pytest.raises(GuardError, match="parse failures"):
            g.check_all(step=1)


class TestDuplicateDetection:
    """Tests for duplicate rule detection."""

    def test_first_embedding_allowed(self):
        """First embedding is always allowed."""
        g = Guards(duplicate_threshold=0.95)
        emb = [1.0, 0.0, 0.0]
        assert g.check_duplicate(emb) is True

    def test_identical_embedding_blocked(self):
        """Identical embedding is blocked."""
        g = Guards(duplicate_threshold=0.95)
        emb = [1.0, 0.0, 0.0]
        g.check_duplicate(emb)
        assert g.check_duplicate(emb) is False

    def test_similar_embedding_blocked(self):
        """Very similar embedding is blocked."""
        g = Guards(duplicate_threshold=0.95)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.99, 0.01, 0.0]  # Very similar
        g.check_duplicate(emb1)
        assert g.check_duplicate(emb2) is False

    def test_different_embedding_allowed(self):
        """Different embedding is allowed."""
        g = Guards(duplicate_threshold=0.95)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]  # Orthogonal
        g.check_duplicate(emb1)
        assert g.check_duplicate(emb2) is True

    def test_bounded_deque(self):
        """Recent embeddings deque is bounded."""
        g = Guards(duplicate_threshold=0.95, max_recent_embeddings=3)

        # Add 4 different embeddings
        g.check_duplicate([1.0, 0.0, 0.0, 0.0])
        g.check_duplicate([0.0, 1.0, 0.0, 0.0])
        g.check_duplicate([0.0, 0.0, 1.0, 0.0])
        g.check_duplicate([0.0, 0.0, 0.0, 1.0])

        # First one should have been evicted (only 3 kept)
        assert len(g.recent_embeddings) == 3

        # First embedding should now be allowed again
        assert g.check_duplicate([1.0, 0.0, 0.0, 0.0]) is True


class TestGuardsInitialization:
    """Tests for Guards initialization."""

    def test_default_values(self):
        """Default values are set correctly."""
        g = Guards()
        assert g.max_cost == 5.0
        assert g.max_rule_ratio == 0.5
        assert g.max_parse_failures == 5
        assert g.warmup == 100
        assert g.duplicate_threshold == 0.95
        assert g.max_recent_embeddings == 10

    def test_custom_values(self):
        """Custom values are set correctly."""
        g = Guards(max_cost=10.0, warmup=50)
        assert g.max_cost == 10.0
        assert g.warmup == 50

    def test_counters_start_at_zero(self):
        """Counters start at zero."""
        g = Guards()
        assert g.samples == 0
        assert g.rules_created == 0
        assert g.consecutive_parse_failures == 0
        assert g.total_cost == 0.0
        assert len(g.recent_embeddings) == 0
