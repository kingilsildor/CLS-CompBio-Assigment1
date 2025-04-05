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


def hill_equation(S: float, v_max: float, K_m: float, n: float) -> float:
    """
    Calculate the rate of a reaction using the Hill equation.

    Params
    ------
    - S (float): Substrate concentration
    - v_max (float): Maximum reaction rate
    - K_m (float): Michaelis constant
    - n (float): Hill coefficient

    Returns
    -------
    - v (float): Reaction rate
    """
    v = (v_max * S**n) / (K_m**n + S**n)
    return v


def bisubstrate_rate(S: tuple, v_max: float, K_m1: float, K_m2: float) -> float:
    """
    Calculate the rate of a reaction using the bisubstrate Michaelis-Menten equation.

    Params
    ------
    - S (tuple): Substrate concentrations (S1, S2)
    - v_max (float): Maximum reaction rate
    - K_m1 (float): Michaelis constant for substrate 1
    - K_m2 (float): Michaelis constant for substrate 2


    Returns
    -------
    - v (float): Reaction rate
    """
    S1, S2 = S
    v = (v_max * S1 * S2) / (K_m1 + S2 + K_m2 * S1 + S1 * S2)
    return v


def r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Calculate the R-squared value for a set of data points.

    Params
    ------
    - y (np.ndarray): Observed data points
    - y_fit (np.ndarray): Fitted data points

    Returns
    -------
    - r2 (float): R-squared value
    """
    ss_res = sum((y - y_fit) ** 2)
    ss_tot = sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def chi_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Calculate the chi-squared value for a set of data points.

    Params
    ------
    - y (np.ndarray): Observed data points
    - y_fit (np.ndarray): Fitted data points

    Returns
    - chi2 (float): Chi-squared value
    -------
    """
    return np.sum(((y - y_fit) ** 2) / y_fit)
