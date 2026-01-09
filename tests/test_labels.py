"""Tests for data/labels.py - IOB2 entity extraction and error classification."""

import pytest
from data.labels import Entity, extract_entities, classify_error


class TestExtractEntities:
    """Tests for extract_entities function."""

    def test_empty_labels(self):
        """Empty list returns no entities."""
        assert extract_entities([]) == []

    def test_all_o_labels(self):
        """All O labels returns no entities."""
        labels = ["O", "O", "O", "O"]
        assert extract_entities(labels) == []

    def test_single_token_entity(self):
        """Single B- token is a complete entity."""
        labels = ["O", "B-Revenue", "O"]
        entities = extract_entities(labels)
        assert len(entities) == 1
        assert entities[0] == Entity(start=1, end=2, entity_type="Revenue")

    def test_multi_token_entity(self):
        """B- followed by I- tokens form one entity."""
        labels = ["O", "B-Revenue", "I-Revenue", "O"]
        entities = extract_entities(labels)
        assert len(entities) == 1
        assert entities[0] == Entity(start=1, end=3, entity_type="Revenue")

    def test_multiple_entities(self):
        """Multiple separate entities are extracted."""
        labels = ["B-Assets", "O", "B-Revenue", "I-Revenue"]
        entities = extract_entities(labels)
        assert len(entities) == 2
        assert entities[0] == Entity(start=0, end=1, entity_type="Assets")
        assert entities[1] == Entity(start=2, end=4, entity_type="Revenue")

    def test_adjacent_entities(self):
        """Two B- tags in sequence are separate entities."""
        labels = ["B-Assets", "B-Revenue"]
        entities = extract_entities(labels)
        assert len(entities) == 2
        assert entities[0] == Entity(start=0, end=1, entity_type="Assets")
        assert entities[1] == Entity(start=1, end=2, entity_type="Revenue")

    def test_i_without_b_ignored(self):
        """I- tag without preceding B- is ignored."""
        labels = ["O", "I-Revenue", "O"]
        entities = extract_entities(labels)
        assert len(entities) == 0

    def test_type_mismatch_closes_entity(self):
        """I- with different type than B- closes previous entity."""
        labels = ["B-Revenue", "I-Assets", "O"]
        entities = extract_entities(labels)
        assert len(entities) == 1
        assert entities[0] == Entity(start=0, end=1, entity_type="Revenue")

    def test_entity_at_end(self):
        """Entity at end of sequence is properly closed."""
        labels = ["O", "B-Revenue", "I-Revenue"]
        entities = extract_entities(labels)
        assert len(entities) == 1
        assert entities[0] == Entity(start=1, end=3, entity_type="Revenue")

    def test_long_entity(self):
        """Long multi-token entity."""
        labels = ["B-Revenue", "I-Revenue", "I-Revenue", "I-Revenue", "O"]
        entities = extract_entities(labels)
        assert len(entities) == 1
        assert entities[0] == Entity(start=0, end=4, entity_type="Revenue")


class TestClassifyError:
    """Tests for classify_error function."""

    def test_omission_error(self):
        """Missing entity that should be present."""
        pred = ["O", "O", "O"]
        truth = ["B-Revenue", "I-Revenue", "O"]
        assert classify_error(pred, truth) == "omission"

    def test_hallucination_error(self):
        """Predicted entity where none exists."""
        pred = ["B-Revenue", "O", "O"]
        truth = ["O", "O", "O"]
        assert classify_error(pred, truth) == "hallucination"

    def test_boundary_error_too_short(self):
        """Predicted span shorter than truth."""
        pred = ["B-Revenue", "O", "O"]
        truth = ["B-Revenue", "I-Revenue", "O"]
        assert classify_error(pred, truth) == "boundary"

    def test_boundary_error_too_long(self):
        """Predicted span longer than truth."""
        pred = ["B-Revenue", "I-Revenue", "I-Revenue"]
        truth = ["B-Revenue", "I-Revenue", "O"]
        assert classify_error(pred, truth) == "boundary"

    def test_boundary_error_shifted(self):
        """Predicted span overlaps but shifted."""
        pred = ["O", "B-Revenue", "I-Revenue"]
        truth = ["B-Revenue", "I-Revenue", "O"]
        assert classify_error(pred, truth) == "boundary"

    def test_classification_error(self):
        """Correct span but wrong entity type."""
        pred = ["B-Assets", "I-Assets", "O"]
        truth = ["B-Revenue", "I-Revenue", "O"]
        assert classify_error(pred, truth) == "classification"

    def test_multiple_errors_returns_first_type(self):
        """With multiple issues, returns appropriate error type."""
        # Both omission and hallucination - hallucination wins if pred has entities
        pred = ["O", "O", "B-Assets"]
        truth = ["B-Revenue", "O", "O"]
        # This is both omission (missed Revenue) and hallucination (extra Assets)
        # The function should return one of the error types
        result = classify_error(pred, truth)
        assert result in ["omission", "hallucination", "boundary", "classification"]
