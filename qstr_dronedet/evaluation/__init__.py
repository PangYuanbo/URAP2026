__all__ = ["classify_error", "evaluate_predictions"]


def __getattr__(name: str):
    if name == "classify_error":
        from .diagnostics import classify_error

        return classify_error
    if name == "evaluate_predictions":
        from .metrics import evaluate_predictions

        return evaluate_predictions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
