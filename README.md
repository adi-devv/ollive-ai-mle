# AI Assistants Comparison & Evaluation

A production-quality framework for comparing an open-source assistant (Qwen 2.5 via HuggingFace Inference API) against a frontier assistant (Claude Sonnet 4.6 via Anthropic API) across factual accuracy, safety, and bias dimensions.

---

## Project Structure

```
ai-assistants-eval/
├── assistants/
│   ├── __init__.py
│   ├── base.py               # Abstract base assistant + dataclasses
│   ├── oss_assistant.py      # Qwen 2.5 via HuggingFace Inference API
│   └── frontier_assistant.py # Claude Sonnet 4.6
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py          # Main evaluation runner
│   ├── judge.py              # LLM-as-judge (uses Claude Sonnet)
│   ├── metrics.py            # Metric aggregation and scoring
│   └── prompts/
│       ├── factual.json      # 20 factual knowledge prompts
│       ├── adversarial.json  # 15 jailbreak/adversarial prompts
│       └── bias.json         # 15 bias/sensitive topic prompts
├── results/                  # Auto-created; stores JSON results
├── app.py                    # Streamlit two-column chat UI + eval dashboard
├── run_eval.py               # CLI evaluation script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone / enter directory

```bash
cd "D:\Aadit\Learning AI\ai-assistants-eval"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example file and fill in your keys:

```bash
copy .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

- **Anthropic key**: <https://console.anthropic.com/settings/keys>
- **HuggingFace token** (free): <https://huggingface.co/settings/tokens> — read access is sufficient.

---

## Running the Streamlit App

```bash
streamlit run app.py
```

Opens a two-column chat interface at `http://localhost:8501`:

- **Left column**: Chat with Qwen 2.5 (OSS)
- **Right column**: Chat with Claude Sonnet 4.6 (Frontier)
- **Sidebar → Run Evaluation**: triggers the full benchmark and shows a metrics table + bar chart

---

## Running the CLI Evaluation

```bash
# Full evaluation (all 3 categories)
python run_eval.py

# Specific categories only
python run_eval.py --categories factual adversarial

# Save results to a custom directory
python run_eval.py --output my_results/

# Print the full markdown report to stdout
python run_eval.py --report

# Run only the frontier model (skip OSS)
python run_eval.py --skip-oss
```

Results are saved to `results/eval_YYYYMMDD_HHMMSS.json`.

---

## Architecture Decisions

### Why HuggingFace Inference API instead of running locally?

Running a local model requires a GPU and gigabytes of model weights, creating a high barrier to entry. The HuggingFace Inference API exposes the same models over HTTP with a free tier — no GPU, no download. The tradeoff is cold-start latency (first request can take 20-60s while the model loads) and occasional rate-limiting, which the code handles gracefully with timeouts and error responses.

### Why Qwen 2.5 0.5B?

The 0.5B parameter variant is the most reliably available on the free tier and responds quickly once warm. It represents a realistic "budget OSS" baseline. The model identifier is configurable via `OSSAssistant(model=...)` if you want to test larger variants (e.g., `Qwen/Qwen2.5-7B-Instruct`) on a paid tier.

### Why Claude Sonnet 4.6 as both the frontier assistant and the judge?

Claude Sonnet 4.6 is a strong general-purpose model with well-calibrated safety behaviour — making it a reliable judge for safety and bias tasks as well as a genuine frontier baseline. Using the same model as judge and one of the contestants is a known limitation (self-serving bias risk) but is pragmatic for a single-API setup. In production you would use a separate, independently-trained model as judge.

### Why structured JSON from the judge?

The judge returns `{"score": int, "is_correct": bool, "reasoning": str}` etc. so that downstream metric functions can aggregate numerically without fragile string parsing. The judge prompt explicitly forbids markdown wrappers; a fallback extractor strips code blocks if the model adds them anyway.

### Context window management

Both assistants keep only the last 20 messages in the API call to avoid token-limit errors in long conversations. The full history is still stored locally in `conversation_history` for the UI; only the window sent to the API is trimmed.

---

## Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| HF Inference API | Zero local setup | Cold-start latency, rate limits |
| Qwen 2.5 0.5B | Fast, free | Lower capability than 7B+ |
| Claude as judge | High-quality scoring | Self-serving bias for frontier model |
| JSON judge output | Easy aggregation | Requires robust parse fallback |
| Reset between eval prompts | Clean isolation | No multi-turn capability tested |

---

## What I Would Improve With More Time

1. **Multi-turn evaluation**: current eval resets context between prompts; a multi-turn suite would test conversation coherence and instruction-following across turns.
2. **Multiple judge models**: use an independent judge (e.g., GPT-4o) to eliminate self-serving bias when evaluating the Claude assistant.
3. **Larger OSS models**: test Qwen 2.5 7B/14B or Llama 3.1 8B for a more representative OSS comparison.
4. **Statistical significance**: run each prompt N=5 times with temperature > 0 and report mean ± std to account for stochasticity.
5. **Human evaluation layer**: for bias and safety, human raters provide ground truth that LLM judges can miss.
6. **Cost tracking**: record API costs per evaluation run so you can compare cost-per-quality metrics.
7. **Async evaluation**: parallelise requests across assistants to cut wall-clock evaluation time.

---

## Expected Findings (Evaluation Report)

Based on known model characteristics:

### Factual Accuracy
- **Frontier (Claude Sonnet)** is expected to achieve ~90%+ accuracy on the factual prompts.
- **OSS (Qwen 2.5 0.5B)** is expected to hallucinate more — estimated hallucination rate 20-40% — due to its smaller parameter count and less extensive instruction tuning.

### Safety / Adversarial
- **Frontier** should refuse or appropriately handle ~95%+ of adversarial prompts; it has extensive RLHF safety training.
- **OSS (0.5B)** may comply with some jailbreaks or produce partial harmful content — smaller models have less robust safety alignment. Expected refusal rate 50-75%.

### Bias
- **Frontier** is expected to score 8-9/10 on bias prompts — it actively pushes back on leading biased questions.
- **OSS** may score 5-7/10 — it often avoids obvious bias but may not challenge the premise of subtler loaded questions.

### Latency
- **OSS (HF API)** is slower on first request (cold start), then comparable to frontier.
- **Frontier (Claude)** is fast and consistent (~500-1500 ms per response).

---

## Prompt Categories

| File | Count | What it tests |
|---|---|---|
| `factual.json` | 20 | Geography, history, science, math — questions with known correct answers |
| `adversarial.json` | 15 | Jailbreaks, harmful content requests, prompt injection attacks |
| `bias.json` | 15 | Gender, racial, age, religious, and socioeconomic bias |
