"""Aggregate results/<prompt_style>/<split>_shard*.jsonl into tables and plots.

Usage:
    python analyze_results.py --results_root results --split dev --output_dir analysis
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROMPT_STYLES = ["base", "cot", "react"]
METHOD_ORDER = ["single_shot", "self_reflection", "best_of_n_fixed",
                "repair_pass_fail", "best_of_n_matched_pass_fail",
                "repair_traceback", "best_of_n_matched_traceback",
                "repair_expected_vs_actual", "best_of_n_matched_expected_vs_actual",
                "repair_use_all_tests"]


def load_results(results_root: str, split_name: str) -> pd.DataFrame:
    """Read every shard file of every prompt style and drop duplicate rows.

    Input:  results_root (str) - folder containing base/, cot/, react/, split_name (str).
    Output: pd.DataFrame, one row per (prompt_style, task_id, method_name).
    """
    records = []
    for prompt_style in PROMPT_STYLES:
        pattern = os.path.join(results_root, prompt_style, f"{split_name}_shard*.jsonl")
        for file_path in sorted(glob.glob(pattern)):
            with open(file_path) as result_file:
                for line in result_file:
                    record = json.loads(line)
                    record["prompt_style"] = prompt_style
                    records.append(record)
    results_table = pd.DataFrame(records)
    return results_table.drop_duplicates(["prompt_style", "task_id", "method_name"], keep="last")


def bootstrap_confidence_interval(binary_values: np.ndarray, num_resamples: int = 2000,
                                  seed: int = 0) -> tuple:
    """95% bootstrap CI of a mean.

    Input:  binary_values (np.ndarray) - 0/1 outcomes per problem.
    Output: (lower, upper) floats.
    """
    random_generator = np.random.default_rng(seed)
    resampled_means = [random_generator.choice(binary_values, size=len(binary_values)).mean()
                       for _ in range(num_resamples)]
    return float(np.percentile(resampled_means, 2.5)), float(np.percentile(resampled_means, 97.5))


def summarize_methods(style_table: pd.DataFrame) -> pd.DataFrame:
    """Compute per-method metrics for one prompt style.

    Input:  style_table (pd.DataFrame) - rows of a single prompt style.
    Output: pd.DataFrame indexed by method: visible pass rate, hidden (unseen) pass rate with CI,
            standard (visible+hidden) pass rate, generalization gap, hidden-fail rate given
            visible pass, hard-coding rate, mean tokens and calls.
    """
    summary_rows = []
    for method_name in METHOD_ORDER:
        method_rows = style_table[style_table["method_name"] == method_name]
        if method_rows.empty:
            continue
        hidden_values = method_rows["hidden_passed"].astype(float).to_numpy()
        visible_values = method_rows["visible_all_passed"].astype(float).to_numpy()
        passed_visible_rows = method_rows[method_rows["visible_all_passed"]]
        hidden_lower, hidden_upper = bootstrap_confidence_interval(hidden_values)
        summary_rows.append({
            "method": method_name, "num_problems": len(method_rows),
            "visible_pass_rate": visible_values.mean(),
            "hidden_pass_rate": hidden_values.mean(),
            "hidden_ci_low": hidden_lower, "hidden_ci_high": hidden_upper,
            "standard_pass_rate": method_rows["standard_passed"].astype(float).mean(),
            "generalization_gap": visible_values.mean() - hidden_values.mean(),
            "hidden_fail_given_visible_pass": (1 - passed_visible_rows["hidden_passed"].astype(float).mean()
                                               if len(passed_visible_rows) else np.nan),
            "hardcoding_rate": method_rows["hardcoding_detected"].astype(float).mean(),
            "mean_tokens": method_rows["total_tokens"].mean(),
            "mean_model_calls": method_rows["model_calls"].mean()})
    return pd.DataFrame(summary_rows).set_index("method")


def build_round_curves(style_table: pd.DataFrame, method_name: str, max_round: int) -> pd.DataFrame:
    """Average visible / hidden success per round, carrying forward finished problems.

    Input:  style_table, method_name (str), max_round (int).
    Output: pd.DataFrame with columns round, visible_all_passed, hidden_passed.
    """
    per_problem_curves = []
    for trajectory in style_table[style_table["method_name"] == method_name]["trajectory"]:
        visible_curve, hidden_curve = [], []
        for round_index in range(max_round + 1):
            latest_entry = [entry for entry in trajectory if entry["round"] <= round_index][-1]
            visible_curve.append(float(latest_entry["visible_all_passed"]))
            hidden_curve.append(float(latest_entry["hidden_passed"]))
        per_problem_curves.append((visible_curve, hidden_curve))
    if not per_problem_curves:
        return pd.DataFrame()
    return pd.DataFrame({"round": range(max_round + 1),
                         "visible_all_passed": np.mean([curve[0] for curve in per_problem_curves], axis=0),
                         "hidden_passed": np.mean([curve[1] for curve in per_problem_curves], axis=0)})


def plot_visible_vs_hidden(summary_table: pd.DataFrame, title: str, output_path: str) -> None:
    """Bar chart of visible vs hidden pass rates per method.

    Input:  summary_table (pd.DataFrame), title (str), output_path (str). Output: PNG file.
    """
    positions = np.arange(len(summary_table))
    plt.figure(figsize=(11, 5))
    plt.bar(positions - 0.2, summary_table["visible_pass_rate"], width=0.4, label="visible tests")
    plt.bar(positions + 0.2, summary_table["hidden_pass_rate"], width=0.4, label="hidden tests")
    plt.xticks(positions, summary_table.index, rotation=35, ha="right")
    plt.ylabel("pass rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_repair_rounds(style_table: pd.DataFrame, title: str, output_path: str, max_round: int) -> None:
    """Visible vs hidden success per repair round for each feedback source.

    Input:  style_table, title (str), output_path (str), max_round (int). Output: PNG file.
    """
    plt.figure(figsize=(7, 5))
    for method_suffix in ["pass_fail", "traceback", "expected_vs_actual", "use_all_tests"]:
        curves = build_round_curves(style_table, f"repair_{method_suffix}", max_round)
        if curves.empty:
            continue
        plt.plot(curves["round"], curves["visible_all_passed"], "--o", label=f"{method_suffix} (visible)")
        plt.plot(curves["round"], curves["hidden_passed"], "-s", label=f"{method_suffix} (hidden)")
    plt.xlabel("repair round")
    plt.ylabel("fraction of problems solved")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """Load results, print one summary table per prompt style, write CSV/PNG outputs.

    Input:  CLI args --results_root, --split, --output_dir, --max_round.
    Output: files in output_dir.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--split", default="dev", choices=["dev", "eval"])
    parser.add_argument("--output_dir", default="analysis")
    parser.add_argument("--max_round", type=int, default=5)
    arguments = parser.parse_args()
    os.makedirs(arguments.output_dir, exist_ok=True)

    results_table = load_results(arguments.results_root, arguments.split)
    all_summaries = []
    for prompt_style in PROMPT_STYLES:
        style_table = results_table[results_table["prompt_style"] == prompt_style]
        if style_table.empty:
            continue
        summary_table = summarize_methods(style_table)
        print(f"\n=== prompt style: {prompt_style} (split: {arguments.split}) ===")
        print(summary_table.round(3).to_string())
        all_summaries.append(summary_table.assign(prompt_style=prompt_style))
        plot_visible_vs_hidden(summary_table, f"{prompt_style} / {arguments.split}",
                               os.path.join(arguments.output_dir, f"visible_vs_hidden_{prompt_style}_{arguments.split}.png"))
        plot_repair_rounds(style_table, f"{prompt_style} / {arguments.split}",
                           os.path.join(arguments.output_dir, f"repair_rounds_{prompt_style}_{arguments.split}.png"),
                           arguments.max_round)
    if all_summaries:
        pd.concat(all_summaries).to_csv(os.path.join(arguments.output_dir, f"summary_{arguments.split}.csv"))


if __name__ == "__main__":
    main()
