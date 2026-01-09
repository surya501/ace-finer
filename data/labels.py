"""IOB2 label utilities for FiNER-139 dataset."""

from dataclasses import dataclass
from functools import lru_cache
from datasets import load_dataset


@lru_cache(maxsize=1)
def get_label_names() -> list[str]:
    """Load and cache the 279 IOB2 label names from FiNER-139."""
    ds = load_dataset("nlpaueb/finer-139", split="train", streaming=True)
    return ds.features["ner_tags"].feature.names


def id_to_label(tag_id: int) -> str:
    """Convert integer tag ID to string label."""
    return get_label_names()[tag_id]


def label_to_id(label: str) -> int:
    """Convert string label to integer tag ID."""
    return get_label_names().index(label)


@dataclass
class Entity:
    """Single extracted entity span."""
    start: int      # Token start index (inclusive)
    end: int        # Token end index (exclusive)
    entity_type: str  # Without B-/I- prefix


def extract_entities(labels: list[str]) -> list[Entity]:
    """
    Extract entity spans from IOB2 labels.

    Example:
        ["O", "B-Revenue", "I-Revenue", "O"] -> [Entity(1, 3, "Revenue")]
    """
    entities = []
    current = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            # Start new entity, close previous if exists
            if current:
                entities.append(current)
            current = Entity(start=i, end=i + 1, entity_type=label[2:])

        elif label.startswith("I-") and current:
            # Continue current entity if type matches
            if label[2:] == current.entity_type:
                current.end = i + 1
            else:
                # Type mismatch, close current and skip this I- tag
                entities.append(current)
                current = None

        else:  # "O" or I- without preceding B-
            if current:
                entities.append(current)
                current = None

    # Don't forget entity at end of sequence
    if current:
        entities.append(current)

    return entities


def classify_error(pred: list[str], truth: list[str]) -> str:
    """
    Classify prediction error type.

    Returns: "boundary" | "classification" | "omission" | "hallucination"
    """
    pred_entities = extract_entities(pred)
    truth_entities = extract_entities(truth)

    pred_spans = {(e.start, e.end) for e in pred_entities}
    truth_spans = {(e.start, e.end) for e in truth_entities}
    pred_types = {(e.start, e.end): e.entity_type for e in pred_entities}
    truth_types = {(e.start, e.end): e.entity_type for e in truth_entities}

    # No truth entities but we predicted some -> hallucination
    if not truth_entities and pred_entities:
        return "hallucination"

    # Truth entities but we predicted none -> omission
    if truth_entities and not pred_entities:
        return "omission"

    # Check for boundary errors (overlapping but not exact spans)
    for ts, te in truth_spans:
        for ps, pe in pred_spans:
            # Overlapping but not exact match
            if (ps < te and pe > ts) and (ps, pe) != (ts, te):
                return "boundary"

    # Exact spans but wrong type -> classification
    for span in pred_spans & truth_spans:
        if pred_types[span] != truth_types[span]:
            return "classification"

    # Default to classification for other mismatches
    return "classification"
