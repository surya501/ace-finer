"""Tests for data/pipeline.py - Sample dataclass and streaming."""

import pytest
from data.pipeline import Sample, stream_finer, batch_samples


class TestSample:
    """Tests for Sample dataclass."""

    def test_sample_fields(self):
        """Sample has all required fields."""
        sample = Sample(
            id=1,
            tokens=["The", "revenue", "was", "$100"],
            ner_labels=["O", "B-Revenue", "O", "O"],
            sentence="The revenue was $100"
        )
        assert sample.id == 1
        assert sample.tokens == ["The", "revenue", "was", "$100"]
        assert sample.ner_labels == ["O", "B-Revenue", "O", "O"]
        assert sample.sentence == "The revenue was $100"

    def test_sample_equality(self):
        """Two samples with same data are equal."""
        s1 = Sample(id=1, tokens=["a"], ner_labels=["O"], sentence="a")
        s2 = Sample(id=1, tokens=["a"], ner_labels=["O"], sentence="a")
        assert s1 == s2


class TestStreamFiner:
    """Tests for stream_finer function."""

    def test_stream_with_limit(self):
        """stream_finer respects limit parameter."""
        samples = list(stream_finer(split="train", limit=5))
        assert len(samples) == 5

    def test_samples_are_sample_objects(self):
        """stream_finer yields Sample objects."""
        samples = list(stream_finer(split="train", limit=1))
        assert len(samples) == 1
        assert isinstance(samples[0], Sample)

    def test_sample_has_string_labels(self):
        """Samples have string ner_labels, not integers."""
        samples = list(stream_finer(split="train", limit=1))
        sample = samples[0]
        for label in sample.ner_labels:
            assert isinstance(label, str)
            assert label == "O" or label.startswith("B-") or label.startswith("I-")

    def test_tokens_and_labels_same_length(self):
        """tokens and ner_labels have same length."""
        samples = list(stream_finer(split="train", limit=10))
        for sample in samples:
            assert len(sample.tokens) == len(sample.ner_labels)

    def test_sentence_is_joined_tokens(self):
        """sentence is space-joined tokens."""
        samples = list(stream_finer(split="train", limit=5))
        for sample in samples:
            assert sample.sentence == " ".join(sample.tokens)


class TestBatchSamples:
    """Tests for batch_samples utility."""

    def test_batch_samples_correct_size(self):
        """batch_samples yields batches of correct size."""
        samples = [Sample(id=i, tokens=[], ner_labels=[], sentence="") for i in range(10)]
        batches = list(batch_samples(iter(samples), batch_size=3))

        assert len(batches) == 4  # 3, 3, 3, 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_batch_samples_preserves_order(self):
        """batch_samples preserves sample order."""
        samples = [Sample(id=i, tokens=[], ner_labels=[], sentence="") for i in range(5)]
        batches = list(batch_samples(iter(samples), batch_size=2))

        all_ids = [s.id for batch in batches for s in batch]
        assert all_ids == [0, 1, 2, 3, 4]

    def test_batch_samples_empty_input(self):
        """batch_samples handles empty input."""
        batches = list(batch_samples(iter([]), batch_size=3))
        assert batches == []

    def test_batch_samples_exact_multiple(self):
        """batch_samples handles input that's exact multiple of batch_size."""
        samples = [Sample(id=i, tokens=[], ner_labels=[], sentence="") for i in range(6)]
        batches = list(batch_samples(iter(samples), batch_size=3))

        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
