"""The five code-generation strategies compared in the study:
single-shot, self-reflection (no execution), best-of-N sampling (fixed N and
token-matched), and test-feedback repair with three feedback levels."""
from dataclasses import dataclass, field
from typing import List, Optional

from src.data_loading import CodingProblem
from src.feedback import build_feedback_text
from src.hardcoding_detection import detect_hardcoding
from src.language_model import GenerationResult, LocalCodeModel
from src.prompts import (build_initial_prompt, build_reflection_prompt,
                         build_repair_prompt, extract_code_block)
from src.sandbox import VisibleTestReport, evaluate_hidden_tests, evaluate_visible_tests


@dataclass
class ExperimentConfig:
    """Hyper-parameters shared by all strategies."""
    max_new_tokens: int = 768
    max_rounds: int = 5                    # repair rounds after the first attempt
    reflection_rounds: int = 2             # self-review rounds without execution
    sampling_temperature: float = 0.8      # used by best-of-N
    max_samples_for_matched_budget: int = 30
    per_test_timeout_seconds: int = 5
    hidden_test_timeout_seconds: int = 60


@dataclass
class TokenUsage:
    """Accumulates compute spent on one problem."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add_generation(self, generation: GenerationResult) -> None:
        """Record the tokens of one model call. Input: GenerationResult. Output: None."""
        self.prompt_tokens += generation.prompt_tokens
        self.completion_tokens += generation.completion_tokens
        self.model_calls += 1

    def copy(self) -> "TokenUsage":
        """Return an independent copy. Input: none. Output: TokenUsage."""
        return TokenUsage(self.prompt_tokens, self.completion_tokens, self.model_calls)


@dataclass
class CandidateEvaluation:
    """Visible + hidden verdicts for one piece of code."""
    code: str
    visible_report: VisibleTestReport
    hidden_passed: bool


@dataclass
class StrategyResult:
    """Everything recorded for one (problem, method) pair."""
    task_id: int
    method_name: str
    final_code: str
    visible_pass_rate: float
    visible_all_passed: bool
    hidden_passed: bool
    total_tokens: int
    model_calls: int
    hardcoded_test_count: int
    hardcoding_detected: bool
    trajectory: List[dict] = field(default_factory=list)
    usage: Optional[TokenUsage] = None     # kept for budget matching; not serialized


def generate_code(model: LocalCodeModel, chat_messages: list, config: ExperimentConfig,
                  usage: TokenUsage, temperature: float) -> str:
    """Call the model once and return the extracted code.

    Input:  model (LocalCodeModel), chat_messages (list), config (ExperimentConfig),
            usage (TokenUsage) - updated in place, temperature (float).
    Output: str - Python source code parsed from the reply.
    """
    generation = model.generate(chat_messages, config.max_new_tokens, temperature)
    usage.add_generation(generation)
    return extract_code_block(generation.text)


def evaluate_candidate(problem: CodingProblem, code: str, config: ExperimentConfig) -> CandidateEvaluation:
    """Run visible and hidden tests on a candidate.

    Input:  problem (CodingProblem), code (str), config (ExperimentConfig).
    Output: CandidateEvaluation. Hidden results are for scoring only and are
            never shown to the model.
    """
    visible_report = evaluate_visible_tests(code, problem.visible_tests, config.per_test_timeout_seconds)
    hidden_passed = evaluate_hidden_tests(code, problem.hidden_test_script, config.hidden_test_timeout_seconds)
    return CandidateEvaluation(code, visible_report, hidden_passed)


def build_trajectory_entry(round_index: int, evaluation: CandidateEvaluation, usage: TokenUsage) -> dict:
    """Summarize one attempt for the per-round curves.

    Input:  round_index (int), evaluation (CandidateEvaluation), usage (TokenUsage) - cumulative.
    Output: dict with round, visible_pass_rate, visible_all_passed, hidden_passed, tokens_so_far.
    """
    return {"round": round_index,
            "visible_pass_rate": evaluation.visible_report.pass_rate,
            "visible_all_passed": evaluation.visible_report.all_passed,
            "hidden_passed": evaluation.hidden_passed,
            "tokens_so_far": usage.total_tokens}


def select_best_by_visible_tests(evaluations: List[CandidateEvaluation]) -> CandidateEvaluation:
    """Pick the candidate with the highest visible pass rate (ties -> latest).

    Input:  evaluations (List[CandidateEvaluation]), non-empty.
    Output: CandidateEvaluation - the selected candidate. Uses only information
            the agent legitimately has (visible tests).
    """
    best_evaluation = evaluations[0]
    for evaluation in evaluations[1:]:
        if evaluation.visible_report.pass_rate >= best_evaluation.visible_report.pass_rate:
            best_evaluation = evaluation
    return best_evaluation


def package_result(problem: CodingProblem, method_name: str, selected: CandidateEvaluation,
                   usage: TokenUsage, trajectory: List[dict]) -> StrategyResult:
    """Assemble a StrategyResult, including the hard-coding check.

    Input:  problem, method_name (str), selected (CandidateEvaluation),
            usage (TokenUsage), trajectory (List[dict]).
    Output: StrategyResult.
    """
    hardcoding_info = detect_hardcoding(selected.code, problem.visible_tests)
    return StrategyResult(
        task_id=problem.task_id, method_name=method_name, final_code=selected.code,
        visible_pass_rate=selected.visible_report.pass_rate,
        visible_all_passed=selected.visible_report.all_passed,
        hidden_passed=selected.hidden_passed, total_tokens=usage.total_tokens,
        model_calls=usage.model_calls,
        hardcoded_test_count=hardcoding_info["hardcoded_test_count"],
        hardcoding_detected=hardcoding_info["hardcoding_detected"],
        trajectory=trajectory, usage=usage.copy())


def run_single_shot(model: LocalCodeModel, problem: CodingProblem, config: ExperimentConfig) -> StrategyResult:
    """Baseline: one greedy generation, no feedback.

    Input:  model, problem, config.
    Output: StrategyResult named 'single_shot'.
    """
    usage = TokenUsage()
    code = generate_code(model, build_initial_prompt(problem), config, usage, temperature=0.0)
    evaluation = evaluate_candidate(problem, code, config)
    return package_result(problem, "single_shot", evaluation, usage,
                          [build_trajectory_entry(0, evaluation, usage)])


def run_self_reflection(model: LocalCodeModel, problem: CodingProblem, config: ExperimentConfig,
                        single_shot_result: StrategyResult) -> StrategyResult:
    """Self-review without executing anything; starts from the single-shot code.

    Input:  model, problem, config, single_shot_result - its code and usage are reused.
    Output: StrategyResult named 'self_reflection' (final = last revision, since
            without execution the model has no signal to pick a better round).
    """
    usage = single_shot_result.usage.copy()
    current_code = single_shot_result.final_code
    trajectory = list(single_shot_result.trajectory)
    final_evaluation = None
    for round_index in range(1, config.reflection_rounds + 1):
        current_code = generate_code(model, build_reflection_prompt(problem, current_code),
                                     config, usage, temperature=0.0)
        final_evaluation = evaluate_candidate(problem, current_code, config)
        trajectory.append(build_trajectory_entry(round_index, final_evaluation, usage))
    return package_result(problem, "self_reflection", final_evaluation, usage, trajectory)


def run_execution_feedback_repair(model: LocalCodeModel, problem: CodingProblem, config: ExperimentConfig,
                                  single_shot_result: StrategyResult, feedback_level: str) -> StrategyResult:
    """Iterative repair using visible-test execution feedback (ReAct-style loop).

    Input:  model, problem, config, single_shot_result - round 0 attempt (reused),
            feedback_level (str) - 'pass_fail' | 'traceback' | 'expected_vs_actual'.
    Output: StrategyResult named 'repair_<feedback_level>'. Stops early when all
            visible tests pass; returns the best candidate by visible pass rate.
    """
    usage = single_shot_result.usage.copy()
    current_code = single_shot_result.final_code
    current_evaluation = evaluate_candidate(problem, current_code, config)
    all_evaluations = [current_evaluation]
    trajectory = [build_trajectory_entry(0, current_evaluation, usage)]

    for round_index in range(1, config.max_rounds + 1):
        if current_evaluation.visible_report.all_passed:
            break
        feedback_text = build_feedback_text(current_evaluation.visible_report, feedback_level)
        repair_messages = build_repair_prompt(problem, current_code, feedback_text)
        current_code = generate_code(model, repair_messages, config, usage, temperature=0.0)
        current_evaluation = evaluate_candidate(problem, current_code, config)
        all_evaluations.append(current_evaluation)
        trajectory.append(build_trajectory_entry(round_index, current_evaluation, usage))

    selected = select_best_by_visible_tests(all_evaluations)
    return package_result(problem, f"repair_{feedback_level}", selected, usage, trajectory)


def run_best_of_n(model: LocalCodeModel, problem: CodingProblem, config: ExperimentConfig,
                  method_name: str, num_samples: Optional[int] = None,
                  token_budget: Optional[int] = None) -> StrategyResult:
    """Sample independent candidates and keep the best on visible tests.

    Input:  model, problem, config, method_name (str),
            num_samples (int, optional)  - fixed number of samples, OR
            token_budget (int, optional) - sample until this many tokens are spent
            (at least one sample is always drawn).
    Output: StrategyResult named `method_name`.
    """
    if num_samples is None and token_budget is None:
        raise ValueError("Provide num_samples or token_budget.")
    usage = TokenUsage()
    all_evaluations: List[CandidateEvaluation] = []
    trajectory: List[dict] = []
    sample_limit = num_samples if num_samples is not None else config.max_samples_for_matched_budget

    while len(all_evaluations) < sample_limit:
        if token_budget is not None and all_evaluations and usage.total_tokens >= token_budget:
            break
        code = generate_code(model, build_initial_prompt(problem), config, usage,
                             temperature=config.sampling_temperature)
        evaluation = evaluate_candidate(problem, code, config)
        all_evaluations.append(evaluation)
        trajectory.append(build_trajectory_entry(len(all_evaluations) - 1, evaluation, usage))

    selected = select_best_by_visible_tests(all_evaluations)
    return package_result(problem, method_name, selected, usage, trajectory)
