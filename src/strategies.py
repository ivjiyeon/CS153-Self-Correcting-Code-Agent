"""The code-generation strategies compared in the study: single-shot,
self-reflection (no execution), best-of-N sampling (fixed N and token-matched),
and test-feedback repair. Repair feedback normally comes from the VISIBLE tests;
the optional `use_all_tests` condition feeds back visible + hidden cases, which
gives an upper bound but makes the hidden score no longer independent."""
from dataclasses import dataclass, field
from typing import List, Optional

from src.data_loading import CodingProblem
from src.feedback import build_feedback_text
from src.hardcoding_detection import detect_hardcoding
from src.language_model import GenerationResult, LocalCodeModel
from src.prompts import (build_initial_prompt, build_reflection_prompt,
                         build_repair_prompt, extract_code_block)
from src.sandbox import TestReport, evaluate_cases


@dataclass
class ExperimentConfig:
    """Hyper-parameters shared by all strategies."""
    prompt_style: str = "react"            # 'base' | 'cot' | 'react'
    max_new_tokens: int = 768
    max_rounds: int = 5                    # repair rounds after the first attempt
    reflection_rounds: int = 2             # self-review rounds without execution
    sampling_temperature: float = 0.8      # used by best-of-N
    max_samples_for_matched_budget: int = 30
    per_case_timeout_seconds: int = 5
    visible_total_timeout_seconds: int = 30
    hidden_total_timeout_seconds: int = 90
    use_all_tests_feedback_level: str = "expected_vs_actual"


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
    """Visible and hidden test reports for one piece of code."""
    code: str
    visible_report: TestReport
    hidden_report: TestReport

    @property
    def hidden_passed(self) -> bool:
        """True if every unseen hidden case passes."""
        return self.hidden_report.all_passed

    @property
    def standard_passed(self) -> bool:
        """EvalPlus-style verdict: all visible AND all hidden cases pass."""
        return self.visible_report.all_passed and self.hidden_report.all_passed

    @property
    def all_tests_report(self) -> TestReport:
        """Visible and hidden outcomes merged (feedback source for use_all_tests)."""
        return TestReport(outcomes=self.visible_report.outcomes + self.hidden_report.outcomes)


@dataclass
class StrategyResult:
    """Everything recorded for one (problem, method) pair."""
    task_id: str
    method_name: str
    final_code: str
    visible_pass_rate: float
    visible_all_passed: bool
    hidden_passed: bool
    standard_passed: bool
    total_tokens: int
    model_calls: int
    hardcoded_test_count: int
    hardcoding_detected: bool
    hidden_tests_used_as_feedback: bool = False
    trajectory: List[dict] = field(default_factory=list)
    usage: Optional[TokenUsage] = None                    # not serialized
    evaluation: Optional[CandidateEvaluation] = None      # not serialized


def generate_code(model: LocalCodeModel, chat_messages: list, config: ExperimentConfig,
                  usage: TokenUsage, temperature: float) -> str:
    """Call the model once and return the extracted code.

    Input:  model, chat_messages (list), config, usage (TokenUsage) - updated in place,
            temperature (float).
    Output: str - Python source parsed from the reply.
    """
    generation = model.generate(chat_messages, config.max_new_tokens, temperature)
    usage.add_generation(generation)
    return extract_code_block(generation.text)


def evaluate_candidate(problem: CodingProblem, code: str, config: ExperimentConfig) -> CandidateEvaluation:
    """Run visible and hidden cases on a candidate.

    Input:  problem (CodingProblem), code (str), config (ExperimentConfig).
    Output: CandidateEvaluation. Hidden results are for scoring only unless the
            use_all_tests condition explicitly feeds them back.
    """
    visible_report = evaluate_cases(code, problem.function_name, problem.visible_cases,
                                    problem.absolute_tolerance, config.per_case_timeout_seconds,
                                    config.visible_total_timeout_seconds)
    hidden_report = evaluate_cases(code, problem.function_name, problem.hidden_cases,
                                   problem.absolute_tolerance, config.per_case_timeout_seconds,
                                   config.hidden_total_timeout_seconds)
    return CandidateEvaluation(code, visible_report, hidden_report)


def build_trajectory_entry(round_index: int, evaluation: CandidateEvaluation, usage: TokenUsage) -> dict:
    """Summarize one attempt for the per-round curves.

    Input:  round_index (int), evaluation (CandidateEvaluation), usage (TokenUsage) - cumulative.
    Output: dict with round, visible/hidden/standard verdicts and tokens so far.
    """
    return {"round": round_index,
            "visible_pass_rate": evaluation.visible_report.pass_rate,
            "visible_all_passed": evaluation.visible_report.all_passed,
            "hidden_passed": evaluation.hidden_passed,
            "standard_passed": evaluation.standard_passed,
            "tokens_so_far": usage.total_tokens}


def select_best_candidate(evaluations: List[CandidateEvaluation], use_all_tests: bool = False
                          ) -> CandidateEvaluation:
    """Pick the candidate with the highest pass rate on the tests the agent may use.

    Input:  evaluations (List[CandidateEvaluation]) non-empty,
            use_all_tests (bool) - rank by visible+hidden pass rate instead of visible only.
    Output: CandidateEvaluation (ties -> latest).
    """
    def pass_rate_seen_by_agent(evaluation: CandidateEvaluation) -> float:
        report = evaluation.all_tests_report if use_all_tests else evaluation.visible_report
        return report.pass_rate

    best_evaluation = evaluations[0]
    for evaluation in evaluations[1:]:
        if pass_rate_seen_by_agent(evaluation) >= pass_rate_seen_by_agent(best_evaluation):
            best_evaluation = evaluation
    return best_evaluation


def package_result(problem: CodingProblem, method_name: str, selected: CandidateEvaluation,
                   usage: TokenUsage, trajectory: List[dict],
                   hidden_tests_used_as_feedback: bool = False) -> StrategyResult:
    """Assemble a StrategyResult, including the hard-coding check.

    Input:  problem, method_name (str), selected (CandidateEvaluation), usage (TokenUsage),
            trajectory (List[dict]), hidden_tests_used_as_feedback (bool).
    Output: StrategyResult.
    """
    hardcoding_info = detect_hardcoding(selected.code, problem.visible_tests)
    return StrategyResult(
        task_id=problem.task_id, method_name=method_name, final_code=selected.code,
        visible_pass_rate=selected.visible_report.pass_rate,
        visible_all_passed=selected.visible_report.all_passed,
        hidden_passed=selected.hidden_passed, standard_passed=selected.standard_passed,
        total_tokens=usage.total_tokens, model_calls=usage.model_calls,
        hardcoded_test_count=hardcoding_info["hardcoded_test_count"],
        hardcoding_detected=hardcoding_info["hardcoding_detected"],
        hidden_tests_used_as_feedback=hidden_tests_used_as_feedback,
        trajectory=trajectory, usage=usage.copy(), evaluation=selected)


def run_single_shot(model: LocalCodeModel, problem: CodingProblem, config: ExperimentConfig) -> StrategyResult:
    """Baseline: one greedy generation, no feedback.

    Input:  model, problem, config.
    Output: StrategyResult named 'single_shot'.
    """
    usage = TokenUsage()
    code = generate_code(model, build_initial_prompt(problem, config.prompt_style), config, usage, 0.0)
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
    final_evaluation = single_shot_result.evaluation
    for round_index in range(1, config.reflection_rounds + 1):
        reflection_messages = build_reflection_prompt(problem, current_code, config.prompt_style)
        current_code = generate_code(model, reflection_messages, config, usage, 0.0)
        final_evaluation = evaluate_candidate(problem, current_code, config)
        trajectory.append(build_trajectory_entry(round_index, final_evaluation, usage))
    return package_result(problem, "self_reflection", final_evaluation, usage, trajectory)


def run_execution_feedback_repair(model: LocalCodeModel, problem: CodingProblem, config: ExperimentConfig,
                                  single_shot_result: StrategyResult, feedback_level: str,
                                  use_all_tests: bool = False) -> StrategyResult:
    """Iterative repair using test-execution feedback.

    Input:  model, problem, config, single_shot_result - round-0 attempt (reused),
            feedback_level (str) - 'pass_fail' | 'traceback' | 'expected_vs_actual',
            use_all_tests (bool) - if True the feedback (and the stopping rule) uses
            visible AND hidden cases; feedback_level is then
            config.use_all_tests_feedback_level.
    Output: StrategyResult named 'repair_<feedback_level>' or 'repair_use_all_tests'.
            Stops early when every test the agent may see passes.
    """
    method_name = "repair_use_all_tests" if use_all_tests else f"repair_{feedback_level}"
    if use_all_tests:
        feedback_level = config.use_all_tests_feedback_level

    usage = single_shot_result.usage.copy()
    current_code = single_shot_result.final_code
    current_evaluation = single_shot_result.evaluation
    all_evaluations = [current_evaluation]
    trajectory = [build_trajectory_entry(0, current_evaluation, usage)]

    for round_index in range(1, config.max_rounds + 1):
        feedback_report = current_evaluation.all_tests_report if use_all_tests else current_evaluation.visible_report
        if feedback_report.all_passed:
            break
        feedback_text = build_feedback_text(feedback_report, feedback_level)
        repair_messages = build_repair_prompt(problem, current_code, feedback_text, config.prompt_style)
        current_code = generate_code(model, repair_messages, config, usage, 0.0)
        current_evaluation = evaluate_candidate(problem, current_code, config)
        all_evaluations.append(current_evaluation)
        trajectory.append(build_trajectory_entry(round_index, current_evaluation, usage))

    selected = select_best_candidate(all_evaluations, use_all_tests)
    return package_result(problem, method_name, selected, usage, trajectory,
                          hidden_tests_used_as_feedback=use_all_tests)


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
        code = generate_code(model, build_initial_prompt(problem, config.prompt_style), config, usage,
                             config.sampling_temperature)
        evaluation = evaluate_candidate(problem, code, config)
        all_evaluations.append(evaluation)
        trajectory.append(build_trajectory_entry(len(all_evaluations) - 1, evaluation, usage))

    selected = select_best_candidate(all_evaluations)
    return package_result(problem, method_name, selected, usage, trajectory)
