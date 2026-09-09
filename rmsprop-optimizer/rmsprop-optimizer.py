import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    new_w = np.array([])
    new_s = np.array([])
    w_arr = np.array(w)
    g_arr = np.array(g)
    s_arr = np.array(s)
    
    new_s = (beta*s_arr) + (1-beta) * np.square(g_arr)
    new_w = w_arr -  lr / np.sqrt(new_s + eps) *  g

    return (new_w,new_s)
    