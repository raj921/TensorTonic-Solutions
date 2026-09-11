import numpy as np

def hinge_loss(
    y_true: list,
    y_score: list,
    margin: float = 1.0,
    reduction: str = "mean"
) -> float:
    """
    Returns the hinge loss as a float.
    """
    y = np.array(y_true)
    ys = np.array(y_score)

    loss = np.maximum(0, margin - y * ys)

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")