"""Turns a VisibleTestReport into the text shown to the model. The three levels
control how much information leaks from the tests (research question 3)."""
from src.sandbox import VisibleTestReport

FEEDBACK_LEVELS = ["pass_fail", "traceback", "expected_vs_actual"]


def build_feedback_text(report: VisibleTestReport, feedback_level: str) -> str:
    """Format execution feedback at the requested level of detail.

    Input:  report (VisibleTestReport) - results of the visible tests,
            feedback_level (str) - 'pass_fail' | 'traceback' | 'expected_vs_actual'.
    Output: str - feedback message for the repair prompt.
    """
    summary_line = f"Your code passed {report.passed_count} of {report.total_count} example tests."
    if feedback_level == "pass_fail":
        return summary_line

    failed_outcomes = [outcome for outcome in report.outcomes if not outcome["passed"]]
    detail_blocks = []
    for outcome in failed_outcomes:
        if feedback_level == "traceback":
            detail_blocks.append(outcome["traceback"] or outcome["error_message"] or "unknown failure")
        elif feedback_level == "expected_vs_actual":
            if outcome["actual"] is not None:
                actual_text = outcome["actual"]
            else:
                actual_text = f"raised {outcome['error_message']}"
            detail_blocks.append(f"Test: {outcome['test']}\nExpected: {outcome['expected']}\nActual: {actual_text}")
        else:
            raise ValueError(f"Unknown feedback level: {feedback_level}")
    return summary_line + "\n\n" + "\n\n".join(detail_blocks)
