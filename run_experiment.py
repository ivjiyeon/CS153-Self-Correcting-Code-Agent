"""Run all strategies on a shard of MBPP+ problems and append results to JSONL.

Example:
    python run_experiment.py --model_name Qwen/Qwen2.5-Coder-3B-Instruct \
        --num_problems 200 --shard_index 0 --num_shards 4
"""
import argparse
import json
import os
from dataclasses import asdict

import torch

from src.data_loading import load_problems
from src.feedback import FEEDBACK_LEVELS
from src.language_model import LocalCodeModel
from src.strategies import (ExperimentConfig, StrategyResult, run_best_of_n,
                            run_execution_feedback_repair, run_self_reflection, run_single_shot)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options. Input: none (sys.argv). Output: argparse.Namespace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dtype_name", default="bfloat16")
    parser.add_argument("--num_problems", type=int, default=200)
    parser.add_argument("--num_visible_tests", type=int, default=3)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--reflection_rounds", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--sampling_temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", default="results")
    return parser.parse_args()


def result_to_json_line(result: StrategyResult) -> str:
    """Serialize a StrategyResult. Input: StrategyResult. Output: one JSON line (str)."""
    result_dictionary = asdict(result)
    result_dictionary.pop("usage", None)
    return json.dumps(result_dictionary)


def load_finished_keys(output_path: str) -> set:
    """Read an existing JSONL file so interrupted jobs can resume.

    Input:  output_path (str).
    Output: set of (task_id, method_name) pairs already stored.
    """
    finished_keys = set()
    if os.path.exists(output_path):
        with open(output_path) as existing_file:
            for line in existing_file:
                record = json.loads(line)
                finished_keys.add((record["task_id"], record["method_name"]))
    return finished_keys


def main() -> None:
    """Run the full comparison on this shard. Input: CLI args. Output: JSONL file in output_dir."""
    arguments = parse_arguments()
    config = ExperimentConfig(max_new_tokens=arguments.max_new_tokens, max_rounds=arguments.max_rounds,
                              reflection_rounds=arguments.reflection_rounds,
                              sampling_temperature=arguments.sampling_temperature)
    os.makedirs(arguments.output_dir, exist_ok=True)
    output_path = os.path.join(arguments.output_dir, f"results_shard{arguments.shard_index}.jsonl")
    finished_keys = load_finished_keys(output_path)

    problems = load_problems(arguments.num_problems, arguments.num_visible_tests, arguments.seed,
                             arguments.shard_index, arguments.num_shards)
    print(f"Shard {arguments.shard_index}/{arguments.num_shards}: {len(problems)} problems", flush=True)
    model = LocalCodeModel(arguments.model_name, arguments.dtype_name)

    expected_method_names = (["single_shot", "self_reflection", "best_of_n_fixed"]
                             + [f"repair_{level}" for level in FEEDBACK_LEVELS]
                             + [f"best_of_n_matched_{level}" for level in FEEDBACK_LEVELS])

    with open(output_path, "a") as output_file:
        for problem_position, problem in enumerate(problems):
            if all((problem.task_id, name) in finished_keys for name in expected_method_names):
                continue
            torch.manual_seed(arguments.seed + problem.task_id)  # reproducible sampling per problem

            problem_results = []
            single_shot_result = run_single_shot(model, problem, config)
            problem_results.append(single_shot_result)
            problem_results.append(run_self_reflection(model, problem, config, single_shot_result))
            problem_results.append(run_best_of_n(model, problem, config, "best_of_n_fixed",
                                                 num_samples=config.max_rounds + 1))
            for feedback_level in FEEDBACK_LEVELS:
                repair_result = run_execution_feedback_repair(model, problem, config,
                                                              single_shot_result, feedback_level)
                problem_results.append(repair_result)
                # Equal-compute control: sample until the repair run's token budget is spent.
                problem_results.append(run_best_of_n(model, problem, config,
                                                     f"best_of_n_matched_{feedback_level}",
                                                     token_budget=repair_result.total_tokens))

            for result in problem_results:
                output_file.write(result_to_json_line(result) + "\n")
            output_file.flush()
            print(f"[{problem_position + 1}/{len(problems)}] task {problem.task_id} done", flush=True)


if __name__ == "__main__":
    main()
