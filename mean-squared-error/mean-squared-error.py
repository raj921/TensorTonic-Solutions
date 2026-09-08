import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    y = np.asarray(y_pred,dtype=float)
    yt = np.asarray(y_true,dtype=float)
    mse = np.mean((yt - y) ** 2)
    return float(mse)
    