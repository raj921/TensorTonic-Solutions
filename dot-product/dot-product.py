import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    t = float(0)
    for i,j in zip(x,y):
        t += i *j
    return t
        