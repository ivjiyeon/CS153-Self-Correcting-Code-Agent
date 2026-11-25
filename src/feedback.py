"""Turns a TestReport into the text shown to the model. The three levels control
how much information leaks from the tests (research question 3)."""
from src.sandbox import TestReport

FEEDBACK_LEVELS = ["pass_fail", "traceback", "expected_vs_actual"]


def build_feedback_text(report: TestReport, feedback_level: str, max_failures_shown: int = 3) -> str:
    """Format execution feedback at the requested level of detail.

    Input:  report (TestReport) - results of the tests the agent may see,
            feedback_level (str) - 'pass_fail' | 'traceback' | 'expected_vs_actual',
            max_failures_shown (int) - cap on failing tests described in detail.
    Output: str - feedback message for the repair prompt.
    """
    summary_line = f"Your code passed {report.passed_count} of {report.total_count} tests."
    if feedback_level == "pass_fail":
        return summary_line

    failed_outcomes = [outcome for outcome in report.outcomes if not outcome["passed"]]
    detail_blocks = []
    for outcome in failed_outcomes[:max_failures_shown]:
        if feedback_level == "traceback":
            detail_blocks.append(outcome["traceback"] or outcome["error_message"] or "unknown failure")
        elif feedback_level == "expected_vs_actual":
            actual_text = outcome["actual"] if outcome["actual"] is not None else f"raised {outcome['error_message']}"
            detail_blocks.append(f"Test: {outcome['test']}\nExpected: {outcome['expected']}\nActual: {actual_text}")
        else:
            raise ValueError(f"Unknown feedback level: {feedback_level}")
    if len(failed_outcomes) > max_failures_shown:
        detail_blocks.append(f"... and {len(failed_outcomes) - max_failures_shown} more failing tests.")
    return summary_line + "\n\n" + "\n\n".join(detail_blocks)
