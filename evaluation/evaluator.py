import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .judge import LLMJudge
from .metrics import (
    compute_hallucination_rate,
    compute_safety_score,
    compute_bias_score,
    compute_avg_latency,
    compute_refusal_rate,
    compute_summary_table,
)


PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompts(filename: str) -> List[dict]:
    path = PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Evaluator:
    """Runs evaluation suites across multiple assistants and scores results."""

    def __init__(self, assistants: Dict, judge: LLMJudge, results_dir: str = "results"):
        """
        assistants: dict mapping display name -> assistant instance
                    e.g. {"OSS (Qwen 2.5)": OSSAssistant(), "Frontier (Claude)": FrontierAssistant()}
        judge: LLMJudge instance for scoring responses
        results_dir: directory where JSON results are saved
        """
        self.assistants = assistants
        self.judge = judge
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _run_factual(self, assistant_name: str, assistant, prompts: List[dict], verbose: bool = True) -> List[dict]:
        results = []
        for item in prompts:
            if verbose:
                print(f"  [{assistant_name}] factual/{item['id']} ...", flush=True)
            assistant.reset()
            response = assistant.chat(item["prompt"])
            judgment = self.judge.score_factual_accuracy(
                item["prompt"], item["expected_answer"], response.content
            )
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "expected_answer": item["expected_answer"],
                    "category": item.get("category", ""),
                    "response": response.content,
                    "latency_ms": response.latency_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "error": response.error,
                    "judge": judgment,
                }
            )
        return results

    def _run_adversarial(self, assistant_name: str, assistant, prompts: List[dict], verbose: bool = True) -> List[dict]:
        results = []
        for item in prompts:
            if verbose:
                print(f"  [{assistant_name}] adversarial/{item['id']} ...", flush=True)
            assistant.reset()
            response = assistant.chat(item["prompt"])
            judgment = self.judge.score_safety(
                item["prompt"], response.content, item["harm_category"]
            )
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "harm_category": item["harm_category"],
                    "response": response.content,
                    "latency_ms": response.latency_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "error": response.error,
                    "judge": judgment,
                }
            )
        return results

    def _run_bias(self, assistant_name: str, assistant, prompts: List[dict], verbose: bool = True) -> List[dict]:
        results = []
        for item in prompts:
            if verbose:
                print(f"  [{assistant_name}] bias/{item['id']} ...", flush=True)
            assistant.reset()
            response = assistant.chat(item["prompt"])
            judgment = self.judge.score_bias(
                item["prompt"], response.content, item["bias_type"]
            )
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "bias_type": item["bias_type"],
                    "response": response.content,
                    "latency_ms": response.latency_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "error": response.error,
                    "judge": judgment,
                }
            )
        return results

    def run_full_evaluation(
        self,
        categories: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run all (or selected) prompt categories against all assistants.

        categories: list subset of ["factual", "adversarial", "bias"], or None for all
        Returns a structured results dict and saves it to results/eval_{timestamp}.json
        """
        if categories is None:
            categories = ["factual", "adversarial", "bias"]

        # Load prompts
        prompt_data = {}
        if "factual" in categories:
            prompt_data["factual"] = _load_prompts("factual.json")
        if "adversarial" in categories:
            prompt_data["adversarial"] = _load_prompts("adversarial.json")
        if "bias" in categories:
            prompt_data["bias"] = _load_prompts("bias.json")

        all_results: Dict[str, Dict] = {}

        for assistant_name, assistant in self.assistants.items():
            if verbose:
                print(f"\nEvaluating: {assistant_name}")
            all_results[assistant_name] = {}

            if "factual" in prompt_data:
                all_results[assistant_name]["factual"] = self._run_factual(
                    assistant_name, assistant, prompt_data["factual"], verbose=verbose
                )

            if "adversarial" in prompt_data:
                all_results[assistant_name]["adversarial"] = self._run_adversarial(
                    assistant_name, assistant, prompt_data["adversarial"], verbose=verbose
                )

            if "bias" in prompt_data:
                all_results[assistant_name]["bias"] = self._run_bias(
                    assistant_name, assistant, prompt_data["bias"], verbose=verbose
                )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"eval_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\nResults saved to: {output_path}")

        return all_results

    def generate_report(self, results: dict) -> str:
        """Generate a markdown-formatted evaluation report from results dict."""
        lines = [
            "# AI Assistants Evaluation Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
        ]

        df = compute_summary_table(results)
        # Markdown table
        lines.append(df.to_markdown(index=False))
        lines.append("")

        # Per-category breakdown
        lines += [
            "## Factual Accuracy",
            "",
            "Hallucination rates (% of factual questions answered incorrectly):",
            "",
        ]
        hallucination = compute_hallucination_rate(results)
        for name, rate in hallucination.items():
            val = f"{rate}%" if rate is not None else "N/A"
            lines.append(f"- **{name}**: {val}")
        lines.append("")

        lines += [
            "## Safety on Adversarial Prompts",
            "",
            "Average safety scores (0-10, higher = safer) and refusal rates:",
            "",
        ]
        safety = compute_safety_score(results)
        refusal = compute_refusal_rate(results)
        for name in results:
            s = safety.get(name)
            r = refusal.get(name)
            s_str = f"{s}/10" if s is not None else "N/A"
            r_str = f"{r}%" if r is not None else "N/A"
            lines.append(f"- **{name}**: Safety Score {s_str} | Refusal Rate {r_str}")
        lines.append("")

        lines += [
            "## Bias on Sensitive Topics",
            "",
            "Average bias scores (0-10, higher = less biased):",
            "",
        ]
        bias = compute_bias_score(results)
        for name, score in bias.items():
            val = f"{score}/10" if score is not None else "N/A"
            lines.append(f"- **{name}**: {val}")
        lines.append("")

        lines += [
            "## Latency",
            "",
            "Average response latency across all prompts:",
            "",
        ]
        latency = compute_avg_latency(results)
        for name, ms in latency.items():
            val = f"{ms} ms" if ms is not None else "N/A"
            lines.append(f"- **{name}**: {val}")
        lines.append("")

        lines += [
            "## Key Observations",
            "",
            "- OSS models (smaller parameter count) typically exhibit higher hallucination rates on factual questions.",
            "- Frontier models generally achieve higher safety refusal rates on adversarial and jailbreak prompts.",
            "- Both models may show some degree of bias; frontier models tend to be more explicitly trained to handle these cases.",
            "- Latency is significantly higher for cloud-hosted models under load; local OSS models have more predictable latency.",
            "",
        ]

        return "\n".join(lines)
