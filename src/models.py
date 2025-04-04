import numpy as np


def michaelis_menten(S: float, v_max: float, K_m: float) -> float:
    """
    Calculate the rate of a reaction using the Michaelis-Menten equation.

    Params
    ------
    - S (float): Substrate concentration
    - v_max (float): Maximum reaction rate
    - K_m (float): Michaelis constant

    Returns
    -------
    - v (float): Reaction rate
    """
    v = (v_max * S) / (K_m + S)
    return v


def r_squared(y: list, y_fit: list) -> float:
    """
    Calculate the R-squared value for a set of data points.

    Params
    ------
    - y (list): Observed data points
    - y_fit (list): Fitted data points

    Returns
    -------
    - r2 (float): R-squared value
    """
    ss_res = sum((y - y_fit) ** 2)
    ss_tot = sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def chi_squared(y: list, y_fit: list) -> float:
    """
    Calculate the chi-squared value for a set of data points.

    Params
    ------
    - y (list): Observed data points
    - y_fit (list): Fitted data points

    Returns
    - chi2 (float): Chi-squared value
    -------
    """
    return np.sum(((y - y_fit) ** 2) / y_fit)
