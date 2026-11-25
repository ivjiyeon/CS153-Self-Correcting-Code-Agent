"""Run all strategies on a shard of one split and append results to JSONL files
in results/<prompt_style>/ (base, cot or react).

Examples:
    python run_experiment.py --prompt_style react --split dev --num_problems 200 --shard_index 0 --num_shards 4
    python run_experiment.py --prompt_style react --split eval --use_all_tests
"""
import argparse
import json
import os
from dataclasses import fields

import torch

from src.data_loading import DEFAULT_DATA_DIRECTORY, SPLIT_NAMES, load_problems
from src.feedback import FEEDBACK_LEVELS
from src.language_model import LocalCodeModel
from src.prompts import PROMPT_STYLES
from src.strategies import (ExperimentConfig, StrategyResult, run_best_of_n,
                            run_execution_feedback_repair, run_self_reflection, run_single_shot)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options. Input: none (sys.argv). Output: argparse.Namespace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dtype_name", default="bfloat16")
    parser.add_argument("--prompt_style", default="react", choices=PROMPT_STYLES)
    parser.add_argument("--split", default="dev", choices=SPLIT_NAMES,
                        help="dev = 90%% of problems for development/main runs; eval = held-out 10%%")
    parser.add_argument("--use_all_tests", action="store_true",
                        help="add the extra condition that feeds back visible AND hidden tests")
    parser.add_argument("--use_all_tests_feedback_level", default="expected_vs_actual", choices=FEEDBACK_LEVELS)
    parser.add_argument("--num_problems", type=int, default=200)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--reflection_rounds", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--sampling_temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--data_directory", default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--results_root", default="results")
    return parser.parse_args()


def result_to_json_line(result: StrategyResult) -> str:
    """Serialize a StrategyResult without its in-memory helper fields.

    Input:  StrategyResult. Output: one JSON line (str).
    """
    skipped_fields = {"usage", "evaluation"}
    serializable_record = {field_info.name: getattr(result, field_info.name)
                           for field_info in fields(result) if field_info.name not in skipped_fields}
    return json.dumps(serializable_record)


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
    """Run the full comparison on this shard. Input: CLI args. Output: JSONL in results/<style>/."""
    arguments = parse_arguments()
    config = ExperimentConfig(prompt_style=arguments.prompt_style, max_new_tokens=arguments.max_new_tokens,
                              max_rounds=arguments.max_rounds, reflection_rounds=arguments.reflection_rounds,
                              sampling_temperature=arguments.sampling_temperature,
                              use_all_tests_feedback_level=arguments.use_all_tests_feedback_level)
    output_directory = os.path.join(arguments.results_root, arguments.prompt_style)
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f"{arguments.split}_shard{arguments.shard_index}.jsonl")
    finished_keys = load_finished_keys(output_path)

    problems = load_problems(arguments.split, arguments.num_problems, arguments.seed,
                             arguments.shard_index, arguments.num_shards, arguments.data_directory)
    print(f"style={arguments.prompt_style} split={arguments.split} shard "
          f"{arguments.shard_index}/{arguments.num_shards}: {len(problems)} problems", flush=True)
    model = LocalCodeModel(arguments.model_name, arguments.dtype_name)

    expected_method_names = (["single_shot", "self_reflection", "best_of_n_fixed"]
                             + [f"repair_{level}" for level in FEEDBACK_LEVELS]
                             + [f"best_of_n_matched_{level}" for level in FEEDBACK_LEVELS]
                             + (["repair_use_all_tests"] if arguments.use_all_tests else []))

    with open(output_path, "a") as output_file:
        for problem_position, problem in enumerate(problems):
            if all((problem.task_id, name) in finished_keys for name in expected_method_names):
                continue
            torch.manual_seed(arguments.seed + problem.task_number)  # reproducible sampling per problem

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
            if arguments.use_all_tests:
                problem_results.append(run_execution_feedback_repair(
                    model, problem, config, single_shot_result, feedback_level="expected_vs_actual",
                    use_all_tests=True))

            for result in problem_results:
                output_file.write(result_to_json_line(result) + "\n")
            output_file.flush()
            print(f"[{problem_position + 1}/{len(problems)}] {problem.task_id} done", flush=True)


if __name__ == "__main__":
    main()
