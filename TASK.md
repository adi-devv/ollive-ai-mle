Build and evaluate two AI personal assistants.
1. Open Source Assistant
Build a simple personal assistant using an open-source model from Hugging Face.
Examples:
Qwen 2.5
Llama 3.2
Phi-3
Mistral
any equivalent OSS model
2. Frontier Model Assistant
Build the same assistant experience with the same capabilities using any hosted foundation model/API.
Examples:
Claude Sonnet
GPT-4.1
Gemini
DeepSeek
Grok
any equivalent
The assistant should support:
multi-turn conversations
short-term conversational memory/context
basic assistant-like behavior
You may use any lightweight interface:
Streamlit
Gradio
FastAPI
Telegram/Discord bot
CLI
web app
any equivalent setup

Evaluation
Compare both assistants on:
Hallucination Rate
Bias & Harmful Outputs (stereotypes, discriminatory behavior, unsafe responses)
Content Safety (jailbreak resistance, refusal handling, robustness to harmful prompts)
You may use:
public benchmarks
custom prompts
LLM-as-judge approaches
your own evaluation framework
We recommend testing with at least:
factual prompts
adversarial/jailbreak prompts
sensitive/bias-related prompts

Deliverables
1. GitHub Repository
Complete source code.
2. README
Include:
setup instructions
architecture decisions
tradeoffs made
what you would improve with more time
3. Short Evaluation Report (1 page)
Include:
comparison results as infographics
recommendations
4. Demo (Optional)
Hosted link, screenshots, or Loom video.

Bonus
You will be given a guaranteed interview if you are able to complete the following task.
Deploy the OSS model publicly
Cost + latency table for the OSS deployment
add observability/evals
add guardrails/safety layers
implement memory/tool use
OSS model can be deployed on any of the following platforms:
Hugging Face Spaces (Qwen2.5-0.5B-Instruct recommended) 
Modal
Ollama
RunPod
Replicate
any equivalent stack
Submission
Please send:
GitHub repo
evaluation pdf
demo link (optional)
to: work@ollive.ai
Looking forward to seeing what you build 🚀
