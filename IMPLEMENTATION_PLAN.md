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
├── main.py                 # Entry point + async loop
├── config.py               # Settings
├── state.py                # State dataclass
│
├── data/
│   ├── __init__.py
│   ├── pipeline.py         # FiNER-139 streaming
│   └── labels.py           # IOB2 label mapping
│
├── playbook/
│   ├── __init__.py
│   ├── schema.py           # Rule dataclass
│   └── store.py            # Chroma wrapper
│
├── agents/
│   ├── __init__.py
│   ├── llm.py              # Groq client with retry/cost tracking
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

### 1. State Dataclass

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
    error_type: str | None = None  # boundary/classification/omission/hallucination

    # Reflection
    new_rule: Rule | None = None

    # Metrics (for logging)
    parse_failed: bool = False
```

### 2. Main Loop

**File**: `main.py`

```python
#!/usr/bin/env python3
import argparse
import asyncio
import logging
from data.pipeline import stream_finer
from playbook.store import PlaybookStore
from agents.generator import Generator
from agents.reflector import Reflector
from agents.curator import Curator
from runner import run_sample
from metrics import MetricsLogger
from guardrails import Guards
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--curation-freq", type=int, default=100)
    p.add_argument("--checkpoint-freq", type=int, default=1000)
    p.add_argument("--max-cost", type=float, default=5.0)
    return p.parse_args()

async def main():
    args = parse_args()
    config = Config()

    # Initialize components
    store = PlaybookStore(path="./playbook_db")
    generator = Generator(config)
    reflector = Reflector(config)
    curator = Curator(store, config)
    metrics = MetricsLogger(window=100)
    guards = Guards(max_cost=args.max_cost)

    step = 0
    for sample in stream_finer(limit=args.max_samples):
        step += 1

        # Process sample
        state = await run_sample(sample, store, generator, reflector, guards)

        # Update metrics
        metrics.record(state)
        guards.check_all(step)

        # Curate periodically
        if step % args.curation_freq == 0:
            report = await curator.run()
            log.info(f"Curation: {report}")

        # Checkpoint periodically
        if step % args.checkpoint_freq == 0:
            store.checkpoint(f"checkpoint_{step}")
            metrics.save(f"metrics_{step}.json")

        # Log progress
        if step % 10 == 0:
            log.info(metrics.summary())

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Runner

**File**: `runner.py`

```python
import asyncio
from state import State
from data.pipeline import Sample
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

    # Initialize state
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

    # 4. Update rule stats (always)
    for rule in state.retrieved_rules:
        store.update_stats(rule.rule_id, success=state.is_correct)

    # 5. Reflect on errors
    if not state.is_correct:
        new_rule = await reflector.generate_rule(state)

        if new_rule and await reflector.validate(new_rule, state, generator):
            if guards.check_duplicate(new_rule, store):
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
    """Process batch with limited concurrency for generator calls."""
    sem = asyncio.Semaphore(max_concurrent)

    async def process(sample):
        async with sem:
            return await run_sample(sample, store, generator, reflector, guards)

    return await asyncio.gather(*[process(s) for s in samples])


def classify_error(pred: list[str], truth: list[str]) -> str:
    """Classify error type: boundary, classification, omission, hallucination."""
    pred_entities = extract_entities(pred)
    truth_entities = extract_entities(truth)

    if not truth_entities and pred_entities:
        return "hallucination"
    if truth_entities and not pred_entities:
        return "omission"
    # More detailed classification...
    return "classification"

def extract_entities(labels: list[str]) -> list[tuple]:
    """Extract (start, end, type) tuples from IOB2 labels."""
    entities = []
    # ... IOB2 parsing logic
    return entities
```

### 4. LLM Client with Retry

**File**: `agents/llm.py`

```python
import asyncio
import openai
import logging
from config import Config

log = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, config: Config):
        self.client = openai.AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=config.groq_api_key
        )
        self.model = config.model
        self.total_cost = 0.0

    async def complete(self, messages: list[dict], retries: int = 3) -> str:
        """Make completion with exponential backoff."""
        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )

                # Track cost
                self.total_cost += self._calc_cost(resp.usage)

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
        # Groq pricing for llama-3.3-70b-versatile
        return (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000
```

### 5. Generator

**File**: `agents/generator.py`

```python
import json
import logging
from agents.llm import LLMClient
from playbook.schema import Rule

log = logging.getLogger(__name__)

class Generator:
    def __init__(self, config):
        self.llm = LLMClient(config)

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
        # XML-tagged tokens
        xml_tokens = " ".join(f'<t id="{i}">{t}</t>' for i, t in enumerate(tokens))

        # Format rules
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
            # Try to extract JSON from response
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

        return labels, True  # All O, parse failed
```

### 6. Guardrails

**File**: `guardrails.py`

```python
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

class GuardError(Exception):
    pass

@dataclass
class Guards:
    max_cost: float = 5.0
    max_rule_ratio: float = 0.5
    max_parse_failures: int = 5
    warmup: int = 100

    # Counters
    samples: int = 0
    rules_created: int = 0
    consecutive_parse_failures: int = 0
    total_cost: float = 0.0
    recent_embeddings: list = None

    def __post_init__(self):
        self.recent_embeddings = []

    def record_sample(self):
        self.samples += 1

    def record_rule_created(self):
        self.rules_created += 1

    def record_parse_failure(self):
        self.consecutive_parse_failures += 1

    def record_parse_success(self):
        self.consecutive_parse_failures = 0

    def record_cost(self, cost: float):
        self.total_cost += cost

    def check_duplicate(self, rule, store) -> bool:
        """Return True if rule is OK to add (not a duplicate)."""
        emb = store.get_embedding(rule.content)
        for recent in self.recent_embeddings[-5:]:
            if cosine_sim(emb, recent) > 0.95:
                log.warning("Duplicate rule detected, skipping")
                return False
        self.recent_embeddings.append(emb)
        return True

    def check_all(self, step: int):
        """Check all guards, raise if violated."""

        # Cost guard
        if self.total_cost > self.max_cost:
            raise GuardError(f"Budget exceeded: ${self.total_cost:.2f} > ${self.max_cost}")

        # Parse failure guard
        if self.consecutive_parse_failures >= self.max_parse_failures:
            raise GuardError(f"{self.max_parse_failures} consecutive parse failures")

        # Rule explosion guard
        if self.samples >= self.warmup:
            ratio = self.rules_created / self.samples
            if ratio > self.max_rule_ratio:
                raise GuardError(f"Rule ratio {ratio:.2f} > {self.max_rule_ratio}")

def cosine_sim(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### 7. Metrics Logger

**File**: `metrics.py`

```python
from collections import deque
from dataclasses import dataclass
import json
import logging

log = logging.getLogger(__name__)

@dataclass
class MetricsLogger:
    window: int = 100

    def __post_init__(self):
        self.history = deque(maxlen=self.window)
        self.total_correct = 0
        self.total_samples = 0
        self.error_counts = {"boundary": 0, "classification": 0, "omission": 0, "hallucination": 0}
        self.parse_failures = 0
        self.rules_created = 0

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
            }, f)
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

## Tests (Write First)

```python
# tests/test_generator.py
def test_parse_valid():
    g = Generator.__new__(Generator)
    labels, failed = g._parse('{"2": "B-Revenue"}', 5)
    assert labels == ["O", "O", "B-Revenue", "O", "O"]
    assert not failed

def test_parse_malformed():
    g = Generator.__new__(Generator)
    labels, failed = g._parse('invalid json', 5)
    assert labels == ["O"] * 5
    assert failed

def test_parse_out_of_bounds():
    g = Generator.__new__(Generator)
    labels, failed = g._parse('{"99": "B-Revenue"}', 5)
    assert labels == ["O"] * 5
    assert not failed  # Parsed OK, just ignored invalid index
```

```python
# tests/test_guardrails.py
def test_cost_guard():
    g = Guards(max_cost=5.0)
    g.total_cost = 6.0
    with pytest.raises(GuardError):
        g.check_all(100)

def test_rule_explosion():
    g = Guards(warmup=10, max_rule_ratio=0.5)
    g.samples = 10
    g.rules_created = 6
    with pytest.raises(GuardError):
        g.check_all(10)
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
uv run python main.py --max-samples 100

# Medium run (primary target)
uv run python main.py --max-samples 10000 --curation-freq 100 --checkpoint-freq 1000
```

---

## What's Removed

| Removed | Why |
|---------|-----|
| LangGraph | Overkill for linear flow |
| langgraph, langchain-core deps | Not needed |
| Graph nodes, state machine | Simple function calls instead |
| Multiple state TypedDicts | Single `State` dataclass |
| Separate monitoring/ dir | Single `guardrails.py` |
| Complex CLI flags | Just 5 essential flags |
| scripts/ directory | Everything in main.py |

---

## Implementation Order

1. `tests/` - Write tests first (mocked)
2. `data/` - Pipeline + labels
3. `playbook/` - Schema + Chroma store
4. `agents/llm.py` - Groq client with retry
5. `agents/generator.py` - Predict with XML tokens
6. `guardrails.py` - Circuit breakers
7. `metrics.py` - Logging
8. `agents/reflector.py` - Rule generation
9. `agents/curator.py` - Dedup/prune
10. `runner.py` - run_sample()
11. `main.py` - Wire it together
