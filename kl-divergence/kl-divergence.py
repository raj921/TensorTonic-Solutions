import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    positive = p > 0

    p_pos = p[positive]
    q_pos = np.clip(q[positive], eps, None)

    return float(np.sum(p_pos * np.log(p_pos / q_pos)))

    
    