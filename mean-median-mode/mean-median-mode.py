from collections import Counter
import numpy as np
from scipy import stats
import numpy as np
from scipy import stats

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    s = {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(stats.mode(x, keepdims=False).mode)
    }

    return s