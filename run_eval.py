"""
CLI evaluation runner.

Usage:
    python run_eval.py
    python run_eval.py --categories factual adversarial bias
    python run_eval.py --categories factual --output results/
    python run_eval.py --skip-oss          # only run frontier
    python run_eval.py --skip-frontier     # only run OSS
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare AI assistants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["factual", "adversarial", "bias"],
        default=["factual", "adversarial", "bias"],
        help="Evaluation categories to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory where JSON results will be saved.",
    )
    parser.add_argument(
        "--skip-oss",
        action="store_true",
        help="Skip the OSS (Qwen 2.5) assistant.",
    )
    parser.add_argument(
        "--skip-frontier",
        action="store_true",
        help="Skip the Frontier (Claude Sonnet) assistant.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the full markdown report after evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate environment ───────────────────────────────────────────────────
    missing = []
    if not args.skip_frontier and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not args.skip_oss and not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN not set. HuggingFace Inference API may rate-limit requests.")

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Set them in a .env file or export them in your shell.")
        sys.exit(1)

    # ── Import and initialise assistants ──────────────────────────────────────
    from assistants import OSSAssistant, FrontierAssistant
    from evaluation import Evaluator, LLMJudge

    assistants = {}

    if not args.skip_oss:
        print("Initialising OSS assistant (Qwen 2.5)…")
        try:
            assistants["OSS (Qwen 2.5)"] = OSSAssistant(hf_token=os.getenv("HF_TOKEN"))
            print("  OK")
        except Exception as e:
            print(f"  Failed: {e}")

    if not args.skip_frontier:
        print("Initialising Frontier assistant (Claude Sonnet 4.6)…")
        try:
            assistants["Frontier (Claude)"] = FrontierAssistant(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            print("  OK")
        except Exception as e:
            print(f"  Failed: {e}")

    if not assistants:
        print("Error: No assistants could be initialised. Exiting.")
        sys.exit(1)

    print(f"\nRunning evaluation on: {list(assistants.keys())}")
    print(f"Categories: {args.categories}")
    print(f"Output dir: {args.output}")
    print("-" * 60)

    # ── Run evaluation ─────────────────────────────────────────────────────────
    judge = LLMJudge(api_key=os.getenv("ANTHROPIC_API_KEY"))
    evaluator = Evaluator(
        assistants=assistants,
        judge=judge,
        results_dir=args.output,
    )

    results = evaluator.run_full_evaluation(
        categories=args.categories,
        verbose=True,
    )

    # ── Print summary table ────────────────────────────────────────────────────
    from evaluation.metrics import compute_summary_table

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    df = compute_summary_table(results)
    print(df.to_string(index=False))
    print()

    # ── Optionally print full report ───────────────────────────────────────────
    if args.report:
        print("\n" + "=" * 60)
        print("FULL REPORT")
        print("=" * 60)
        report = evaluator.generate_report(results)
        print(report)


if __name__ == "__main__":
    main()
