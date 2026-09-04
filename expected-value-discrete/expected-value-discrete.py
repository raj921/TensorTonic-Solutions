import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    w = np.array(x)
    e = np.array(p)

    return float(np.sum(w * e))