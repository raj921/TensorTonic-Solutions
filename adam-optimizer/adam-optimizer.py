import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    # Update first moment
    m_new = beta1 * m + (1 - beta1) * grad

    # Update second moment
    v_new = beta2 * v + (1 - beta2) * np.square(grad)

    # Bias correction
    mt = m_new / (1 - beta1**t)
    vt = v_new / (1 - beta2**t)

    # Parameter update
    param_new = param - lr * mt / (np.sqrt(vt) + eps)

    return (param_new, m_new, v_new)
    






    


    
