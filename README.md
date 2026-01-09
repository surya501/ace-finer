# ACE-FiNER: Agentic Context Engineering for XBRL Tagging

**ACE-FiNER** is an implementation of an autonomous agentic system designed to master Named Entity Recognition (NER) on financial documents (FiNER-139 dataset).

Instead of fine-tuning a model, this system uses **Reflexion** to fine-tune a "Playbook" of instructions. It learns from its mistakes by generating and storing semantic rules, which are retrieved and injected into the prompt context for future similar tasks.

## Results

After training on just 100 samples, ACE dramatically improves accuracy on held-out data:

| Metric | Baseline (no rules) | With ACE Rules |
|--------|---------------------|----------------|
| Accuracy | 38.0% | **92.0%** |
| Correct samples | 19/50 | 46/50 |

**Improvement: +54% absolute, +142% relative**

The learned rules generalize well - in testing, rules helped in 27 cases and hurt in 0 cases.

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Groq](https://groq.com/) API Key

### Installation

1.  **Clone and Setup**:
    ```bash
    git clone <repository-url>
    cd ace-finer
    uv venv
    source .venv/bin/activate
    uv sync
    ```

2.  **Environment**:
    Create a `.env` file in the root directory:
    ```bash
    GROQ_API_KEY=gsk_your_key_here
    ```

### Usage

**Run a small batch (Test Drive):**
```bash
uv run python main.py --max-samples 50 --batch-size 5
```

**Run a full training loop:**
```bash
uv run python main.py --max-samples 1000 --curation-freq 100
```

**CLI Options:**
- `--max-samples`: Maximum samples to process (default: all)
- `--batch-size`: Batch size for processing (default: 20)
- `--max-concurrent`: Maximum concurrent API calls (default: 10)
- `--curation-freq`: Run curation every N samples (default: 100)
- `--checkpoint-freq`: Checkpoint every N samples (default: 1000)
- `--max-cost`: Maximum API cost in dollars (default: 5.0)
- `--model`: Groq model to use (default: openai/gpt-oss-120b)

---

## How It Works

### The Core Philosophy
> "Don't just retrain the model; improve the instructions it receives."

This system implements a **Reflexion**-style loop where:
1.  A **Student (Generator)** tries to solve a problem
2.  A **Teacher (Reflector)** critiques mistakes and writes a new "Rule"
3.  A **Librarian (Playbook)** stores this rule in a semantic database
4.  On future similar problems, the system "remembers" this rule

### Architecture

```
Input Sample → Retrieve Rules → Generator → Evaluate
                                    ↓
                              Is Correct?
                             /          \
                          Yes            No
                           ↓              ↓
                     Update Stats    Reflector
                                         ↓
                                   Generate Rule
                                         ↓
                                   Validate Rule
                                    /        \
                               Valid        Invalid
                                 ↓              ↓
                          Store in         Discard
                          Playbook
```

### Core Components

| Component | Role | File |
|-----------|------|------|
| **Playbook** | Vector database storing learned rules | `playbook/store.py` |
| **Generator** | The "Student" - performs NER tagging | `agents/generator.py` |
| **Reflector** | The "Teacher" - analyzes errors, creates rules | `agents/reflector.py` |
| **Curator** | The "Librarian" - prunes bad/duplicate rules | `agents/curator.py` |
| **Guards** | Circuit breakers for cost, quality, duplicates | `guardrails.py` |

### The Playbook (Long-Term Memory)

The Playbook is a Vector Database (ChromaDB) that stores **Rules**. Instead of finetuning the LLM, we finetune the *database* of rules.

- **Storage**: Rules are embedded using `sentence-transformers`
- **Retrieval**: For every new sentence, query for top-k semantically similar rules
- **Evolution**: Rules track `success_count` and `failure_count`. Bad rules are pruned by the Curator

### Dynamic Prompting

The Generator's prompt is **dynamic** - it changes based on retrieved rules:

```text
System: You are an expert financial tagger.
Context Rules:
- Rule 1: "When you see 'net revenue', tag it as Revenue."
- Rule 2: "Don't tag percentages as monetary values."
Input: "The net revenue was 50% higher."
Task: Tag tokens.
```

By changing the "Context Rules", the Generator's behavior changes without changing the code or model.

### Validation-Driven Memory

We never trust the Reflector blindly. A rule is only saved if it is **proven** (via `reflector.validate()`) to fix the specific error that triggered its creation. This prevents "hallucinated" or ineffective rules from polluting the database.

---

## Project Structure

```
ace-finer/
├── main.py                 # Entry point, batch loop
├── runner.py               # Orchestrates single-sample lifecycle
├── state.py                # State dataclass passed between agents
├── metrics.py              # Streaming accuracy, logging
├── guardrails.py           # Circuit breakers
│
├── agents/
│   ├── llm.py              # Groq client with retry/cost callback
│   ├── generator.py        # The "Student" (tags text)
│   ├── reflector.py        # The "Teacher" (analyzes errors)
│   └── curator.py          # The "Librarian" (cleans up rules)
│
├── playbook/
│   ├── schema.py           # Rule dataclass
│   └── store.py            # ChromaDB wrapper
│
├── data/
│   ├── pipeline.py         # FiNER-139 streaming + Sample dataclass
│   └── labels.py           # IOB2 label mapping + entity extraction
│
└── tests/                  # 69 unit tests
```

---

## Key Design Patterns

### "In-Context Learning" as a Database
Instead of relying on the LLM's implicit training data, we make knowledge **explicit** and **retrievable**. This allows us to "patch" the model's behavior instantly by adding a rule, rather than waiting for a retraining cycle.

### Self-Refining Guards
The `guardrails.py` ensures the agent doesn't go off the rails:
- **Cost Guards**: Stops execution if API costs exceed a budget
- **Quality Guards**: Stops if the agent creates too many rules (overfitting) or fails parsing repeatedly
- **Duplicate Guards**: Prevents adding rules that are vector-similar to existing ones

### Error Classification
The system classifies prediction errors into four types:
- **Omission**: Failed to tag an entity that exists
- **Hallucination**: Tagged something that shouldn't be an entity
- **Boundary**: Got the span wrong (too short/long)
- **Classification**: Right span, wrong entity type

---

## Experiments

Key metrics to watch in the logs:
- **`rules_created`**: How many new rules is the system inventing?
- **`window_acc`**: Is the accuracy over the last 100 samples improving?
- **`parse_fails`**: Is the LLM struggling to output valid JSON?

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment logs.

---

## Development

This project uses **bd** (beads) for issue tracking:

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
```

Run tests:
```bash
uv run pytest tests/ -v
```

---

## License

MIT
