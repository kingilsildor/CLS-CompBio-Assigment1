import numpy as np
from scipy.optimize import curve_fit

from src.models import bisubstrate_rate, hill_equation, michaelis_menten


def hill_curve(S: np.ndarray, v: np.ndarray) -> tuple:
    """
    Fit the Hill equation curve to the data.

    Params
    ------
    - S (np.ndarray): Substrate concentrations
    - v (np.ndarray): Reaction rates

    Returns
    -------
    - params (tuple): Fitted parameters (v_max, K_m, n)
    - covariance (np.ndarray): Covariance of the fitted parameters
    """
    params, covariance = curve_fit(
        hill_equation, S, v, bounds=([0.1, 0.1, 0.1], [1.1, 1.1, 1.1])
    )
    assert len(params) == 3, "Expected 3 parameters for Hill curve fitting"
    return params, covariance


def michaelis_menten_curve(S: np.ndarray, v: np.ndarray) -> tuple:
    """
    Fit the Michaelis-Menten equation curve to the data.

    Params
    ------
    - S (np.ndarray): Substrate concentrations
    - v (np.ndarray): Reaction rates

    Returns
    -------
    - params (tuple): Fitted parameters (v_max, K_m)
    - covariance (np.ndarray): Covariance of the fitted parameters
    """
    params, covariance = curve_fit(
        michaelis_menten, S, v, bounds=([0.1, 0.1], [1.1, 1.1])
    )
    assert len(params) == 2, "Expected 2 parameters for Michaelis-Menten curve fitting"
    return params, covariance


def bisubstrate_curve(S1: np.ndarray, S2: np.ndarray, v: np.ndarray) -> tuple:
    """
    Fit the bisubstrate Michaelis-Menten equation curve to the data.

    Params
    ------
    - S1 (np.ndarray): Substrate 1 concentrations
    - S2 (np.ndarray): Substrate 2 concentrations
    - v (np.ndarray): Reaction rates

    Returns
    -------
    - params (tuple): Fitted parameters (v_max, K_m1, K_m2)
    - covariance (np.ndarray): Covariance of the fitted parameters
    """
    params, covariance = curve_fit(
        bisubstrate_rate,
        (S1, S2),
        v,
        bounds=([0.1, 0.1, 0.1], [1.1, 1.1, 1.1]),
    )
    assert len(params) == 3, "Expected 3 parameters for bisubstrate curve fitting"
    return params, covariance
