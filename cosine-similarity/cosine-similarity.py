import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here

    v1 = np.array(a)
    v2 = np.array(b)

    dot_prod = np.dot(v1, v2)
    norm_x = np.linalg.norm(v1)
    norm_y = np.linalg.norm(v2)

    if norm_x == 0 or norm_y == 0:
        return 0.0
    similarity = dot_prod / (norm_x * norm_y)
    return float(similarity)
    

    