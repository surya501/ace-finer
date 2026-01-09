"""Data pipeline for FiNER-139 dataset."""

from dataclasses import dataclass
from typing import Iterator
from datasets import load_dataset
from data.labels import id_to_label


@dataclass
class Sample:
    """Single FiNER-139 sample with normalized field names."""
    id: int
    tokens: list[str]
    ner_labels: list[str]  # IOB2 string labels (converted from ner_tags)
    sentence: str


def stream_finer(split: str = "train", limit: int | None = None) -> Iterator[Sample]:
    """
    Stream FiNER-139 samples.

    Converts integer ner_tags to string ner_labels.

    Args:
        split: Dataset split ("train", "validation", "test")
        limit: Maximum number of samples to yield

    Yields:
        Sample objects with string labels
    """
    dataset = load_dataset("nlpaueb/finer-139", split=split, streaming=True)

    count = 0
    for record in dataset:
        if limit and count >= limit:
            break

        # Convert integer tags to string labels
        ner_labels = [id_to_label(tag) for tag in record["ner_tags"]]

        yield Sample(
            id=record["id"],
            tokens=record["tokens"],
            ner_labels=ner_labels,
            sentence=" ".join(record["tokens"])
        )
        count += 1


def batch_samples(samples_iter: Iterator[Sample], batch_size: int) -> Iterator[list[Sample]]:
    """
    Yield batches of samples.

    Args:
        samples_iter: Iterator of Sample objects
        batch_size: Maximum size of each batch

    Yields:
        Lists of Sample objects, each of size batch_size (except possibly last)
    """
    batch = []
    for sample in samples_iter:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
