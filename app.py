"""
Streamlit UI for comparing OSS (Qwen 2.5) and Frontier (Claude Sonnet 4.6) assistants.
Run with: streamlit run app.py
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistants Comparison",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Assistants Comparison")
st.caption("Comparing **Llama 3.1 8B** (OSS · small model) vs **Llama 3.3 70B** (Frontier · large model) — both via Groq")


# ── Lazy-load assistants (cached so they persist across reruns) ────────────────
@st.cache_resource(show_spinner="Initialising assistants…")
def load_assistants():
    from assistants import OSSAssistant, FrontierAssistant

    oss = OSSAssistant(api_key=os.getenv("GROQ_API_KEY"))
    frontier = FrontierAssistant(api_key=os.getenv("GROQ_API_KEY"))
    return oss, frontier


try:
    oss_assistant, frontier_assistant = load_assistants()
    assistants_ready = True
except Exception as e:
    assistants_ready = False
    st.error(f"Failed to initialise assistants: {e}")

# ── Session state ──────────────────────────────────────────────────────────────
if "oss_messages" not in st.session_state:
    st.session_state.oss_messages = []          # [{role, content, meta}]
if "frontier_messages" not in st.session_state:
    st.session_state.frontier_messages = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_running" not in st.session_state:
    st.session_state.eval_running = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    st.subheader("OSS Assistant")
    oss_model = st.text_input("Model", value="llama-3.1-8b-instant", key="oss_model")
    st.caption("Small open-source model (8B params) via Groq — free tier")

    st.divider()
    st.subheader("Frontier Assistant")
    st.caption("llama-3.3-70b-versatile via Groq API")

    st.divider()
    st.subheader("Evaluation")

    eval_categories = st.multiselect(
        "Categories to evaluate",
        options=["factual", "adversarial", "bias"],
        default=["factual", "adversarial", "bias"],
    )

    run_eval_btn = st.button(
        "Run Evaluation",
        disabled=not assistants_ready or st.session_state.eval_running,
        use_container_width=True,
        type="primary",
    )

    st.divider()
    st.caption("Set `GROQ_API_KEY` in `.env`")


# ── Run evaluation ─────────────────────────────────────────────────────────────
if run_eval_btn and assistants_ready:
    st.session_state.eval_running = True

    with st.spinner("Running evaluation… this may take several minutes."):
        from evaluation import Evaluator, LLMJudge

        judge = LLMJudge(api_key=os.getenv("GROQ_API_KEY"))
        evaluator = Evaluator(
            assistants={
                "OSS (Llama 3.1 8B)": oss_assistant,
                "Frontier (Claude)": frontier_assistant,
            },
            judge=judge,
            results_dir="results",
        )
        results = evaluator.run_full_evaluation(
            categories=eval_categories if eval_categories else None,
            verbose=False,
        )
        st.session_state.eval_results = results

    st.session_state.eval_running = False
    st.success("Evaluation complete!")
    st.rerun()


# ── Display evaluation results ────────────────────────────────────────────────
if st.session_state.eval_results:
    from evaluation.metrics import compute_summary_table

    st.header("Evaluation Results")
    df = compute_summary_table(st.session_state.eval_results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart: key metrics side by side
    chart_df = df.set_index("Assistant")[
        [c for c in df.columns if c != "Assistant" and df[c].notna().any()]
    ]
    # Normalise Latency (ms) to 0-10 scale for visual comparison
    if "Avg Latency (ms)" in chart_df.columns:
        max_latency = chart_df["Avg Latency (ms)"].max()
        if max_latency and max_latency > 0:
            chart_df["Avg Latency (ms)"] = (
                chart_df["Avg Latency (ms)"] / max_latency * 10
            ).round(2)
            chart_df = chart_df.rename(
                columns={"Avg Latency (ms)": "Avg Latency (normalised /10)"}
            )

    st.subheader("Metric Comparison (all metrics on 0-10 or % scale)")
    st.bar_chart(chart_df.T)

    # Download evaluation report
    from evaluation import Evaluator, LLMJudge

    judge = LLMJudge(api_key=os.getenv("GROQ_API_KEY"))
    evaluator = Evaluator(
        assistants={},
        judge=judge,
        results_dir="results",
    )
    report_md = evaluator.generate_report(st.session_state.eval_results)
    st.download_button(
        "Download Full Report (Markdown)",
        data=report_md,
        file_name="evaluation_report.md",
        mime="text/markdown",
    )

    st.divider()


# ── Chat columns ───────────────────────────────────────────────────────────────
st.header("Chat Interface")

col_oss, col_frontier = st.columns(2)

# ── OSS column ─────────────────────────────────────────────────────────────────
with col_oss:
    st.subheader("OSS Assistant")
    st.caption(f"Model: `{oss_model}` | Provider: Groq (free tier)")

    # Clear button
    if st.button("Clear conversation", key="clear_oss"):
        st.session_state.oss_messages = []
        if assistants_ready:
            oss_assistant.reset()
        st.rerun()

    # Chat history
    chat_container_oss = st.container(height=450)
    with chat_container_oss:
        for msg in st.session_state.oss_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns(3)
                    cols[0].caption(f"Latency: {meta.get('latency_ms', '?')} ms")
                    cols[1].caption(f"In: {meta.get('input_tokens', '?')} tok")
                    cols[2].caption(f"Out: {meta.get('output_tokens', '?')} tok")

    # Input
    oss_input = st.chat_input("Message Llama 3.1 8B…", key="oss_input")
    if oss_input and assistants_ready:
        st.session_state.oss_messages.append({"role": "user", "content": oss_input})
        with st.spinner("Llama 3.1 8B thinking…"):
            resp = oss_assistant.chat(oss_input)
        st.session_state.oss_messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "meta": {
                    "latency_ms": resp.latency_ms,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            }
        )
        st.rerun()


# ── Frontier column ────────────────────────────────────────────────────────────
with col_frontier:
    st.subheader("Frontier Assistant")
    st.caption("Model: `llama-3.3-70b-versatile` | Provider: Groq")

    # Clear button
    if st.button("Clear conversation", key="clear_frontier"):
        st.session_state.frontier_messages = []
        if assistants_ready:
            frontier_assistant.reset()
        st.rerun()

    # Chat history
    chat_container_frontier = st.container(height=450)
    with chat_container_frontier:
        for msg in st.session_state.frontier_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns(3)
                    cols[0].caption(f"Latency: {meta.get('latency_ms', '?')} ms")
                    cols[1].caption(f"In: {meta.get('input_tokens', '?')} tok")
                    cols[2].caption(f"Out: {meta.get('output_tokens', '?')} tok")

    # Input
    frontier_input = st.chat_input("Message Claude Sonnet…", key="frontier_input")
    if frontier_input and assistants_ready:
        st.session_state.frontier_messages.append(
            {"role": "user", "content": frontier_input}
        )
        with st.spinner("Claude thinking…"):
            resp = frontier_assistant.chat(frontier_input)
        st.session_state.frontier_messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "meta": {
                    "latency_ms": resp.latency_ms,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            }
        )
        st.rerun()
