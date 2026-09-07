import numpy as np
import numpy as np

def sigmoid(x: list | float | np.ndarray) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar, list, or NumPy array.
    """
    arr = np.asarray(x, dtype=float)
    result = 1 / (1 + np.exp(-arr))
    
    
    return float(result) if np.isscalar(x) else result
        