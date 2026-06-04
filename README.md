# ollive-ai-mle — AI Assistants Comparison & Evaluation

A framework for comparing an open-source assistant (Qwen 2.5 via HuggingFace Inference API) against a frontier assistant (Llama 3.3 70B via Groq) across factual accuracy, safety, and bias dimensions.

**OSS model:** Qwen/Qwen2.5-0.5B-Instruct via HuggingFace Inference API  
**Frontier model:** llama-3.3-70b-versatile via Groq API  
**Judge:** llama-3.3-70b-versatile via Groq API (LLM-as-judge)

---

## Project Structure

```
ollive-ai-mle/
├── assistants/
│   ├── base.py               # Abstract base assistant + dataclasses
│   ├── oss_assistant.py      # Qwen 2.5 via HuggingFace Inference API
│   └── frontier_assistant.py # Llama 3.3 70B via Groq
├── evaluation/
│   ├── evaluator.py          # Main evaluation runner
│   ├── judge.py              # LLM-as-judge (Groq)
│   ├── metrics.py            # Metric aggregation
│   └── prompts/
│       ├── factual.json      # 20 factual knowledge prompts
│       ├── adversarial.json  # 15 jailbreak/adversarial prompts
│       └── bias.json         # 15 bias/sensitive topic prompts
├── results/                  # Auto-created; stores JSON results
├── app.py                    # Streamlit two-column chat UI + eval dashboard
├── run_eval.py               # CLI evaluation script
├── evaluation_report.md      # Pre-run evaluation report with findings
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your-groq-key-here
HF_TOKEN=your-hf-token-here   # optional — public models work without a token
```

- **Groq key** (free): https://console.groq.com
- **HuggingFace token** (optional, free): https://huggingface.co/settings/tokens

### 3. Run the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Running the CLI Evaluation

```bash
# Full evaluation (all 3 categories)
python run_eval.py

# Specific categories only
python run_eval.py --categories factual adversarial

# Print markdown report to stdout
python run_eval.py --report

# Skip OSS model (run frontier only)
python run_eval.py --skip-oss
```

Results saved to `results/eval_YYYYMMDD_HHMMSS.json`.

---

## Evaluation Categories

| File | Count | Tests |
|------|-------|-------|
| `factual.json` | 20 | Geography, history, science, math — known correct answers |
| `adversarial.json` | 15 | Jailbreaks, harmful content requests, prompt injection |
| `bias.json` | 15 | Gender, racial, age, religious, socioeconomic bias |

**Scoring method:** LLM-as-judge (Llama 3.3 70B via Groq) scores each response 0–10 on:
- Factual accuracy (`is_correct`, `score`)
- Safety (`refused`, `score`)
- Bias (`bias_detected`, `score`)

---

## Architecture Decisions

### Why HuggingFace Inference API instead of running locally?
Running a local model needs a GPU and gigabytes of weights. The HF Inference API exposes the same models over HTTP on a free tier — no GPU needed. Tradeoff: cold-start latency (20–60s first request) and rate limits. Code handles both with timeouts and graceful error responses.

### Why Qwen 2.5 0.5B?
The 0.5B variant is the most reliably available on the HF free tier. It represents a realistic "budget OSS" baseline. The model is configurable — pass `model="Qwen/Qwen2.5-7B-Instruct"` for a stronger comparison.

### Why Groq for the frontier model?
Groq provides fast, free-tier inference on Llama 3.3 70B — a genuinely capable frontier-class open model. Using Groq avoids paid API costs while still demonstrating the frontier vs OSS capability gap.

### Why the same model as judge and frontier assistant?
Using Llama 3.3 70B as both frontier assistant and judge is a known limitation (self-serving bias risk) but is pragmatic for a single-API setup. In production, use an independently-trained model as judge (e.g., a different model family).

### Why structured JSON from the judge?
The judge returns `{"score": int, "is_correct": bool, "reasoning": str}` so downstream metrics can aggregate numerically without fragile string parsing. A fallback extractor strips markdown code fences if the model adds them.

---

## Tradeoffs

| Decision | Benefit | Cost |
|----------|---------|------|
| HF Inference API | Zero local setup | Cold-start latency, rate limits |
| Qwen 2.5 0.5B | Fast, free | Lower capability than 7B+ |
| Groq as judge | Free, fast | Self-serving bias for frontier model |
| JSON judge output | Easy aggregation | Requires robust parse fallback |
| Reset between eval prompts | Clean isolation | No multi-turn capability tested |

---

## What I'd Improve with More Time

1. **Deploy OSS model publicly** — HuggingFace Spaces or Modal for a live endpoint
2. **Multiple judge models** — use a separate model family to eliminate self-serving bias
3. **Larger OSS models** — Qwen 2.5 7B/14B or Llama 3.1 8B for a fairer comparison
4. **Statistical significance** — run each prompt N=5 times, report mean ± std
5. **Guardrails layer** — add Llama Guard or keyword filtering as a safety pre-screen
6. **Memory + tool use** — conversation summarization + calculator/web search tool
7. **Cost tracking** — log token usage per query, compute cost-per-quality metrics
8. **Async evaluation** — parallelise requests to cut wall-clock evaluation time

---

## Expected Findings

See [`evaluation_report.md`](./evaluation_report.md) for detailed results and recommendations.

| Metric | OSS (Qwen 2.5 0.5B) | Frontier (Llama 3.3 70B) |
|--------|---------------------|--------------------------|
| Hallucination Rate | ~35–45% | ~5–10% |
| Safety Score (0–10) | ~6.5 | ~9.5+ |
| Refusal Rate | ~55% | ~90%+ |
| Bias Score (0–10) | ~6.8 | ~9.0+ |
| Avg Latency | ~3,000–5,000ms | ~800–1,500ms |

