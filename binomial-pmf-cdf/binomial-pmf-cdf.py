import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with PMF and CDF.
    """

    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")

    if not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    if not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("k must be an integer between 0 and n")

    pmf = float(
        math.comb(n, k)
        * (p ** k)
        * ((1 - p) ** (n - k))
    )

    cdf = float(sum(
        math.comb(n, i)
        * (p ** i)
        * ((1 - p) ** (n - i))
        for i in range(k + 1)
    ))

    return {
        "pmf": pmf,
        "cdf": cdf
    }