# Experiment Log

Use this file to track your runs, hypothesis, and results. Since ACE-FiNER learns dynamically, the "Model" is effectively the combination of the **LLM + The Playbook**.

---

## 2026-01-08 - A/B Test: Baseline vs ACE Rules

**Goal**: Measure how much ACE improves accuracy over baseline (no rules).

**Method**:
1. Train on 100 samples (samples 1-100) to build playbook
2. Test on 50 held-out samples (samples 101-150)
3. Compare: baseline (no rules) vs with ACE rules

**Configuration**:
- `model`: openai/gpt-oss-120b (via Groq)
- `training_samples`: 100
- `test_samples`: 50 (held-out)
- `batch_size`: 20
- `embedding_model`: all-MiniLM-L6-v2

**Training Results**:
- Rules created: 21
- Rules pruned (duplicates): 2
- Training accuracy: 53%
- Parse failures: 13
- Cost: $0.20

**A/B Test Results**:

| Metric | Baseline | With ACE Rules |
|--------|----------|----------------|
| Accuracy | 38.0% | **92.0%** |
| Correct | 19/50 | 46/50 |

- **Absolute improvement**: +54%
- **Relative improvement**: +142%
- Rules helped: 27 cases
- Rules hurt: 0 cases

**Sample Rules Generated**:
1. "Do not tag references to accounting standards (e.g., 'ASC 842') as entities"
2. "When you see a percentage following 'interest rate', tag it as DebtInstrumentInterestRate"
3. "Include the '%' symbol as part of the interest rate span"
4. "When a $ amount is labeled 'Conversion Price', use ConversionPrice not SharePrice"

**Observations**:
- Rules generalize well to unseen samples
- First 20 held-out samples: 100% accuracy with rules vs 30% baseline
- Common error patterns (hallucination on ASC references) were learned quickly

---

## 2026-01-08 - Model Comparison: llama-3.3-70b vs gpt-oss-120b

**Goal**: Compare Llama 3.3 70B vs GPT OSS 120B on 100 samples.

**Configuration**:
- `max_samples`: 100
- `batch_size`: 10

**Results**:

| Metric | llama-3.3-70b-versatile | openai/gpt-oss-120b |
|--------|-------------------------|---------------------|
| Accuracy | 45% | **50%** |
| Parse failures | 18 | **6** |
| Rules created | 9 | 9 |
| Cost | $0.17 | $0.20 |

**Observations**:
- GPT OSS 120B has significantly fewer parse failures (6 vs 18)
- Slightly higher accuracy (50% vs 45%)
- Both create similar number of rules
- GPT OSS 120B is marginally more expensive

**Recommendation**: Use `openai/gpt-oss-120b` for better JSON parsing reliability.

---

## Template

### Experiment ID: [Date]-[Name]
- **Goal**: (e.g., "Test if Llama-3-70b creates better rules than Mixtral")
- **Configuration**:
  - `model`: (e.g. openai/gpt-oss-120b)
  - `max_samples`:
  - `batch_size`:
  - `embedding_model`: (default: all-MiniLM-L6-v2)
- **Starting State**: (Empty Playbook / Pre-filled Playbook v1)
- **Results**:
  - Final Accuracy:
  - Total Rules Created:
  - Total Cost:
- **Observations**:
  - (e.g., "The model struggled with 'Operating Income' vs 'Net Income' until rule #45 was created.")
