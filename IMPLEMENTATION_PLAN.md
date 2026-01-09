# ACE Framework - Simplified Implementation Plan

## Design Principles

1. **No LangGraph** - Simple async loop with function calls
2. **Single state dataclass** - Pass through functions
3. **Minimal dependencies** - Only what's needed
4. **Linear and debuggable** - Easy to trace execution

---

## Project Structure

```
mohit_ace/
├── pyproject.toml
├── .env                    # GROQ_API_KEY
├── main.py                 # Entry point + batch loop
├── config.py               # Settings
├── state.py                # State dataclass
│
├── data/
│   ├── __init__.py
│   ├── pipeline.py         # FiNER-139 streaming + Sample dataclass
│   └── labels.py           # IOB2 label mapping + entity extraction
│
├── playbook/
│   ├── __init__.py
│   ├── schema.py           # Rule dataclass
│   └── store.py            # Chroma wrapper (full API)
│
├── agents/
│   ├── __init__.py
│   ├── llm.py              # Groq client with retry/cost callback
│   ├── generator.py        # Predict tags
│   ├── reflector.py        # Generate rules
│   └── curator.py          # Dedup/prune
│
├── runner.py               # run_sample(), run_batch()
├── metrics.py              # Streaming F1, logging
├── guardrails.py           # Circuit breakers
│
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_playbook.py
    ├── test_generator.py
    └── test_guardrails.py
```

---

## Core Components

### 1. Sample Dataclass (Data Pipeline)

**File**: `data/pipeline.py`

```python
from dataclasses import dataclass
from datasets import load_dataset
from data.labels import id_to_label

@dataclass
class Sample:
    """Single FiNER-139 sample with normalized field names."""
    id: int
    tokens: list[str]
    ner_labels: list[str]  # IOB2 string labels (converted from ner_tags)
    sentence: str

def stream_finer(split: str = "train", limit: int | None = None):
    """
    Stream FiNER-139 samples.
    Converts integer ner_tags to string ner_labels.
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

def batch_samples(samples_iter, batch_size: int):
    """Yield batches of samples."""
    batch = []
    for sample in samples_iter:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

### 2. IOB2 Entity Extraction

**File**: `data/labels.py`

```python
from functools import lru_cache
from datasets import load_dataset

@lru_cache(maxsize=1)
def get_label_names() -> list[str]:
    """Load and cache the 279 IOB2 label names."""
    ds = load_dataset("nlpaueb/finer-139", split="train", streaming=True)
    return ds.features["ner_tags"].feature.names

def id_to_label(tag_id: int) -> str:
    return get_label_names()[tag_id]

def label_to_id(label: str) -> int:
    return get_label_names().index(label)

@dataclass
class Entity:
    """Single extracted entity span."""
    start: int      # Token start index
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
            # Start new entity
            if current:
                entities.append(current)
            current = Entity(start=i, end=i+1, entity_type=label[2:])

        elif label.startswith("I-") and current:
            # Continue current entity if type matches
            if label[2:] == current.entity_type:
                current.end = i + 1
            else:
                # Type mismatch, close current and skip
                entities.append(current)
                current = None

        else:  # "O" or I- without B-
            if current:
                entities.append(current)
                current = None

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
            # Overlapping but not exact
            if (ps < te and pe > ts) and (ps, pe) != (ts, te):
                return "boundary"

    # Exact spans but wrong type -> classification
    for span in pred_spans & truth_spans:
        if pred_types[span] != truth_types[span]:
            return "classification"

    # Default to classification for other mismatches
    return "classification"
```

### 3. State Dataclass

**File**: `state.py`

```python
from dataclasses import dataclass, field
from playbook.schema import Rule

@dataclass
class State:
    """Passed through each step of processing."""
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
    error_type: str | None = None

    # Reflection
    new_rule: Rule | None = None

    # Metrics
    parse_failed: bool = False
```

### 4. PlaybookStore API

**File**: `playbook/store.py`

```python
import chromadb
from chromadb.utils import embedding_functions
from dataclasses import asdict
from datetime import datetime
from playbook.schema import Rule
import json
import shutil

class PlaybookStore:
    """
    ChromaDB-backed rule storage with embedding search.
    All methods are synchronous (Chroma is sync).
    """

    def __init__(self, path: str = "./playbook_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.path = path
        self.client = chromadb.PersistentClient(path=path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="rules",
            embedding_function=self.embed_fn
        )

    def add_rule(self, rule: Rule) -> None:
        """Insert a new rule. Skips if rule_id already exists."""
        existing = self.collection.get(ids=[rule.rule_id])
        if existing["ids"]:
            return  # Already exists

        self.collection.add(
            ids=[rule.rule_id],
            documents=[rule.content],
            metadatas=[rule.to_metadata()]
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Rule]:
        """Find top-k rules by semantic similarity to query."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        rules = []
        for i, rule_id in enumerate(results["ids"][0]):
            rules.append(Rule.from_query_result(
                rule_id=rule_id,
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            ))
        return rules

    def update_stats(self, rule_id: str, success: bool) -> None:
        """Increment success or failure count for a rule."""
        existing = self.collection.get(ids=[rule_id])
        if not existing["ids"]:
            return

        metadata = existing["metadatas"][0]
        if success:
            metadata["success_count"] = metadata.get("success_count", 0) + 1
        else:
            metadata["failure_count"] = metadata.get("failure_count", 0) + 1
        metadata["last_used"] = datetime.utcnow().isoformat()

        self.collection.update(ids=[rule_id], metadatas=[metadata])

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text (for duplicate detection)."""
        return self.embed_fn([text])[0]

    def get_all_rules(self) -> list[Rule]:
        """Retrieve all rules (for curation)."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get()
        rules = []
        for i, rule_id in enumerate(results["ids"]):
            rules.append(Rule.from_query_result(
                rule_id=rule_id,
                content=results["documents"][i],
                metadata=results["metadatas"][i]
            ))
        return rules

    def delete_rule(self, rule_id: str) -> None:
        """Delete a rule by ID."""
        self.collection.delete(ids=[rule_id])

    def count(self) -> int:
        """Total number of rules."""
        return self.collection.count()

    def checkpoint(self, name: str) -> None:
        """Copy current DB to checkpoint directory."""
        checkpoint_path = f"./checkpoints/{name}"
        shutil.copytree(self.path, checkpoint_path, dirs_exist_ok=True)
```

**File**: `playbook/schema.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Rule:
    rule_id: str
    content: str
    trigger_context: str
    target_entities: list[str]
    error_type: str = "classification"
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used: str | None = None

    @staticmethod
    def create(content: str, trigger_context: str, target_entities: list[str], error_type: str) -> "Rule":
        return Rule(
            rule_id=str(uuid.uuid4()),
            content=content,
            trigger_context=trigger_context,
            target_entities=target_entities,
            error_type=error_type
        )

    def to_metadata(self) -> dict:
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
        return cls(
            rule_id=rule_id,
            content=content,
            trigger_context=metadata.get("trigger_context", ""),
            target_entities=metadata.get("target_entities", "").split(","),
            error_type=metadata.get("error_type", "classification"),
            success_count=metadata.get("success_count", 0),
            failure_count=metadata.get("failure_count", 0),
            created_at=metadata.get("created_at", ""),
            last_used=metadata.get("last_used") or None
        )

    def utility_score(self) -> float:
        """Laplace-smoothed success rate."""
        return (self.success_count + 1) / (self.success_count + self.failure_count + 2)
```

### 5. LLM Client with Cost Callback

**File**: `agents/llm.py`

```python
import asyncio
import openai
import logging
from typing import Callable

log = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        on_cost: Callable[[float], None] | None = None  # Cost callback
    ):
        self.client = openai.AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.model = model
        self.on_cost = on_cost  # Called after each request with cost

    async def complete(self, messages: list[dict], retries: int = 3) -> str:
        """Make completion with exponential backoff."""
        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )

                # Calculate and report cost
                cost = self._calc_cost(resp.usage)
                if self.on_cost:
                    self.on_cost(cost)

                return resp.choices[0].message.content

            except openai.RateLimitError:
                wait = 2 ** attempt
                log.warning(f"Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            except openai.APIError as e:
                log.warning(f"API error: {e}, retry {attempt+1}/{retries}")
                await asyncio.sleep(1)

        raise RuntimeError("LLM call failed after retries")

    def _calc_cost(self, usage) -> float:
        # Groq pricing for llama-3.3-70b-versatile ($/1M tokens)
        return (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000
```

### 6. Generator

**File**: `agents/generator.py`

```python
import json
import logging
from agents.llm import LLMClient
from playbook.schema import Rule

log = logging.getLogger(__name__)

class Generator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def predict(self, tokens: list[str], rules: list[Rule]) -> tuple[list[str], bool]:
        """
        Predict IOB2 labels for tokens.
        Returns: (labels, parse_failed)
        """
        prompt = self._build_prompt(tokens, rules)
        response = await self.llm.complete([{"role": "user", "content": prompt}])

        labels, failed = self._parse(response, len(tokens))
        if failed:
            log.warning(f"Parse failed: {response[:100]}...")

        return labels, failed

    def _build_prompt(self, tokens: list[str], rules: list[Rule]) -> str:
        xml_tokens = " ".join(f'<t id="{i}">{t}</t>' for i, t in enumerate(tokens))
        rules_text = "\n".join(f"- {r.content}" for r in rules) if rules else "(none)"

        return f"""Tag tokens with XBRL entity labels (IOB2 format).

Rules:
{rules_text}

Tokens:
{xml_tokens}

Output JSON mapping token IDs to non-O labels, e.g. {{"3": "B-Revenue"}}
Return {{}} if all tokens are O."""

    def _parse(self, response: str, n: int) -> tuple[list[str], bool]:
        """Parse JSON response, return (labels, parse_failed)."""
        labels = ["O"] * n

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                for idx_str, label in data.items():
                    idx = int(idx_str)
                    if 0 <= idx < n:
                        labels[idx] = label
                return labels, False
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"JSON parse error: {e}")

        return labels, True
```

### 7. Guardrails (with bounded deque)

**File**: `guardrails.py`

```python
from collections import deque
from dataclasses import dataclass, field
import logging
import numpy as np

log = logging.getLogger(__name__)

class GuardError(Exception):
    pass

@dataclass
class Guards:
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
        self.samples += 1

    def record_rule_created(self):
        self.rules_created += 1

    def record_parse_failure(self):
        self.consecutive_parse_failures += 1

    def record_parse_success(self):
        self.consecutive_parse_failures = 0

    def record_cost(self, cost: float):
        """Called by LLM client after each request."""
        self.total_cost += cost

    def check_duplicate(self, embedding: list[float]) -> bool:
        """
        Check if embedding is too similar to recent ones.
        Returns True if OK to add, False if duplicate.
        """
        emb = np.array(embedding)
        for recent in self.recent_embeddings:
            sim = np.dot(emb, recent) / (np.linalg.norm(emb) * np.linalg.norm(recent))
            if sim > self.duplicate_threshold:
                log.warning(f"Duplicate rule detected (sim={sim:.3f}), skipping")
                return False

        self.recent_embeddings.append(emb)
        return True

    def check_all(self, step: int):
        """Check all guards, raise if violated."""
        if self.total_cost > self.max_cost:
            raise GuardError(f"Budget exceeded: ${self.total_cost:.2f} > ${self.max_cost}")

        if self.consecutive_parse_failures >= self.max_parse_failures:
            raise GuardError(f"{self.max_parse_failures} consecutive parse failures")

        if self.samples >= self.warmup:
            ratio = self.rules_created / self.samples
            if ratio > self.max_rule_ratio:
                raise GuardError(f"Rule ratio {ratio:.2f} > {self.max_rule_ratio}")
```

### 8. Runner

**File**: `runner.py`

```python
import asyncio
from state import State
from data.pipeline import Sample
from data.labels import classify_error
from playbook.store import PlaybookStore
from agents.generator import Generator
from agents.reflector import Reflector
from guardrails import Guards

async def run_sample(
    sample: Sample,
    store: PlaybookStore,
    generator: Generator,
    reflector: Reflector,
    guards: Guards
) -> State:
    """Process one sample through the full pipeline."""

    state = State(
        sample_id=sample.id,
        sentence=sample.sentence,
        tokens=sample.tokens,
        ground_truth=sample.ner_labels
    )

    # 1. Retrieve rules
    state.retrieved_rules = store.retrieve(state.sentence, top_k=5)

    # 2. Generate predictions
    state.predictions, state.parse_failed = await generator.predict(
        state.tokens,
        state.retrieved_rules
    )

    if state.parse_failed:
        guards.record_parse_failure()
    else:
        guards.record_parse_success()

    # 3. Evaluate
    state.is_correct = (state.predictions == state.ground_truth)
    if not state.is_correct:
        state.error_type = classify_error(state.predictions, state.ground_truth)

    # 4. Update rule stats
    for rule in state.retrieved_rules:
        store.update_stats(rule.rule_id, success=state.is_correct)

    # 5. Reflect on errors
    if not state.is_correct:
        new_rule = await reflector.generate_rule(state)

        if new_rule:
            # Validate rule fixes the error
            if await reflector.validate(new_rule, state, generator):
                # Check for duplicates
                embedding = store.get_embedding(new_rule.content)
                if guards.check_duplicate(embedding):
                    store.add_rule(new_rule)
                    state.new_rule = new_rule
                    guards.record_rule_created()

    guards.record_sample()
    return state


async def run_batch(
    samples: list[Sample],
    store: PlaybookStore,
    generator: Generator,
    reflector: Reflector,
    guards: Guards,
    max_concurrent: int = 10
) -> list[State]:
    """Process batch with limited concurrency."""
    sem = asyncio.Semaphore(max_concurrent)

    async def process(sample):
        async with sem:
            return await run_sample(sample, store, generator, reflector, guards)

    return await asyncio.gather(*[process(s) for s in samples])
```

### 9. Main Loop (with batching)

**File**: `main.py`

```python
#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
from dotenv import load_dotenv

from data.pipeline import stream_finer, batch_samples
from playbook.store import PlaybookStore
from agents.llm import LLMClient
from agents.generator import Generator
from agents.reflector import Reflector
from agents.curator import Curator
from runner import run_batch
from metrics import MetricsLogger
from guardrails import Guards, GuardError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--curation-freq", type=int, default=100)
    p.add_argument("--checkpoint-freq", type=int, default=1000)
    p.add_argument("--max-cost", type=float, default=5.0)
    return p.parse_args()

async def main():
    args = parse_args()

    # Initialize guards first (need cost callback)
    guards = Guards(max_cost=args.max_cost)

    # Initialize LLM with cost callback
    llm = LLMClient(
        api_key=os.environ["GROQ_API_KEY"],
        on_cost=guards.record_cost  # Wire cost tracking
    )

    # Initialize components
    store = PlaybookStore(path="./playbook_db")
    generator = Generator(llm)
    reflector = Reflector(llm)
    curator = Curator(store, llm)
    metrics = MetricsLogger(window=100)

    step = 0
    samples_iter = stream_finer(limit=args.max_samples)

    try:
        for batch in batch_samples(samples_iter, args.batch_size):
            # Process batch concurrently
            states = await run_batch(
                batch, store, generator, reflector, guards,
                max_concurrent=args.max_concurrent
            )

            # Update metrics and check guards
            for state in states:
                step += 1
                metrics.record(state)
                guards.check_all(step)

            # Curate periodically
            if step % args.curation_freq < args.batch_size:
                report = await curator.run()
                log.info(f"Curation: {report}")

            # Checkpoint periodically
            if step % args.checkpoint_freq < args.batch_size:
                store.checkpoint(f"checkpoint_{step}")
                metrics.save(f"metrics_{step}.json")

            # Log progress
            log.info(metrics.summary())

    except GuardError as e:
        log.error(f"Guard triggered: {e}")
        store.checkpoint("checkpoint_error")
        metrics.save("metrics_error.json")
        raise

    except KeyboardInterrupt:
        log.info("Interrupted, saving checkpoint...")
        store.checkpoint("checkpoint_interrupted")
        metrics.save("metrics_interrupted.json")

if __name__ == "__main__":
    asyncio.run(main())
```

### 10. Metrics Logger

**File**: `metrics.py`

```python
from collections import deque
from dataclasses import dataclass, field
import json

@dataclass
class MetricsLogger:
    window: int = 100
    history: deque = field(default=None, init=False)
    total_correct: int = field(default=0, init=False)
    total_samples: int = field(default=0, init=False)
    error_counts: dict = field(default=None, init=False)
    parse_failures: int = field(default=0, init=False)
    rules_created: int = field(default=0, init=False)

    def __post_init__(self):
        self.history = deque(maxlen=self.window)
        self.error_counts = {"boundary": 0, "classification": 0, "omission": 0, "hallucination": 0}

    def record(self, state):
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
        window_acc = sum(self.history) / len(self.history) if self.history else 0
        total_acc = self.total_correct / self.total_samples if self.total_samples else 0

        return (
            f"step={self.total_samples} | "
            f"window_acc={window_acc:.1%} | "
            f"total_acc={total_acc:.1%} | "
            f"rules={self.rules_created} | "
            f"parse_fails={self.parse_failures}"
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "total_samples": self.total_samples,
                "total_correct": self.total_correct,
                "error_counts": self.error_counts,
                "parse_failures": self.parse_failures,
                "rules_created": self.rules_created
            }, f, indent=2)
```

---

## Dependencies

**File**: `pyproject.toml`

```toml
[project]
name = "ace-finer"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "datasets>=2.14.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "openai>=1.0.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.23.0", "pytest-mock>=3.12.0"]
```

---

## Tests

```python
# tests/test_labels.py
from data.labels import extract_entities, classify_error, Entity

def test_extract_single_entity():
    labels = ["O", "B-Revenue", "I-Revenue", "O"]
    entities = extract_entities(labels)
    assert len(entities) == 1
    assert entities[0] == Entity(1, 3, "Revenue")

def test_extract_multiple_entities():
    labels = ["B-Assets", "O", "B-Revenue", "I-Revenue"]
    entities = extract_entities(labels)
    assert len(entities) == 2

def test_classify_omission():
    pred = ["O", "O", "O"]
    truth = ["B-Revenue", "I-Revenue", "O"]
    assert classify_error(pred, truth) == "omission"

def test_classify_hallucination():
    pred = ["B-Revenue", "O", "O"]
    truth = ["O", "O", "O"]
    assert classify_error(pred, truth) == "hallucination"

def test_classify_boundary():
    pred = ["B-Revenue", "O", "O"]
    truth = ["B-Revenue", "I-Revenue", "O"]
    assert classify_error(pred, truth) == "boundary"
```

```python
# tests/test_generator.py
from agents.generator import Generator

def test_parse_valid():
    g = Generator.__new__(Generator)
    labels, failed = g._parse('{"2": "B-Revenue"}', 5)
    assert labels == ["O", "O", "B-Revenue", "O", "O"]
    assert not failed

def test_parse_malformed():
    g = Generator.__new__(Generator)
    labels, failed = g._parse('invalid', 5)
    assert labels == ["O"] * 5
    assert failed
```

```python
# tests/test_guardrails.py
import pytest
from guardrails import Guards, GuardError

def test_cost_guard():
    g = Guards(max_cost=5.0)
    g.record_cost(6.0)
    with pytest.raises(GuardError):
        g.check_all(100)

def test_duplicate_detection():
    g = Guards()
    emb = [1.0, 0.0, 0.0]
    assert g.check_duplicate(emb) == True  # First one OK
    assert g.check_duplicate(emb) == False  # Duplicate
```

---

## Execution

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync
echo "GROQ_API_KEY=gsk_xxx" > .env

# Test first
uv run pytest tests/ -v

# Small run
uv run python main.py --max-samples 100 --batch-size 10

# Medium run (primary target)
uv run python main.py --max-samples 10000 --batch-size 50 --curation-freq 100
```

---

## Fixes Applied

| Issue | Fix |
|-------|-----|
| `--batch-size` unused | Wired through `batch_samples()` + `run_batch()` |
| `Sample` undefined | Defined in `data/pipeline.py` with correct fields |
| `PlaybookStore` API undefined | Full API specified: `add_rule`, `retrieve`, `update_stats`, `get_embedding`, `checkpoint` |
| Cost tracking split | `LLMClient` takes `on_cost` callback, wired to `guards.record_cost` |
| `classify_error` placeholder | Full IOB2 span extraction + boundary/classification/omission/hallucination logic |
| `recent_embeddings` unbounded | Uses `deque(maxlen=10)`, embeddings passed directly (not re-fetched) |
