import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    p = y_pred[np.arange(len(y_true)), y_true]
    loss = -np.mean(np.log(p))

    return float(loss)
    