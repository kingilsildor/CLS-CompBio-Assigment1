import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import *


def plot_lineweaver_burk(
    df: pd.DataFrame,
    v_max: float,
    k_m: float,
    max_value: float = 1.0,
    save: bool = False,
) -> None:
    """
    Plot the Lineweaver-Burk plot for the given DataFrame.

    Params
    ------
    - df (pd.DataFrame): DataFrame containing the data to plot.
    - v_max (float): Maximum reaction rate.
    - k_m (float): Michaelis constant.
    - max_value (float): Maximum value for S2 to consider in the plot. Default is 1.0.
    - save (bool): Whether to save the plot as an image file. Default is False.
    """
    plt.figure(figsize=(10, 6))
    for s2, group in df[df["S2"] <= max_value].groupby("S2"):
        inv_S1 = 1 / group["S1"]
        inv_Rate = 1 / group["Rate"]
        plt.plot(inv_S1, inv_Rate, "o-", label=f"1/[S2] = {1 / s2:.1f} mM⁻¹")

    plt.xlabel("1/[S1] (mM⁻¹)")
    plt.ylabel("1/Rate (s/mM)")
    plt.title(
        f"Lineweaver-Burk Plot for different values of [S2]\n$K_m$ = {k_m:.2f} mM, $v_{{max}}$ = {v_max:.2f} mM/s"
    )
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "lineweaver_burk_plot.png"), dpi=FIG_DPI)
    else:
        plt.show()


def plot_eadie_hofstee(
    df: pd.DataFrame,
    v_max: float,
    k_m: float,
    max_value: float = 1.0,
    save: bool = False,
) -> None:
    """
    Plot the Eadie-Hofstee plot for the given DataFrame.

    Params
    ------
    - df (pd.DataFrame): DataFrame containing the data to plot.
    - v_max (float): Maximum reaction rate.
    - k_m (float): Michaelis constant.
    - max_value (float): Maximum value for S2 to consider in the plot. Default is 1.0.
    - save (bool): Whether to save the plot as an image file. Default is False.
    """
    plt.figure(figsize=(10, 6))
    for s2, group in df[df["S2"] <= max_value].groupby("S2"):
        v = group["Rate"]
        s1 = group["S1"]
        plt.plot(v, v / s1, "o-", label=f"[S2] = {s2:.1f} mM")

    plt.xlabel("Rate (mM/s)")
    plt.ylabel("Rate/[S1] (s/mM)")
    plt.title(
        f"Eadie-Hofstee Plot for different values of [S2]\n$K_m$ = {k_m:.2f} mM, $v_{{max}}$ = {v_max:.2f} mM/s"
    )
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "eadie_hofstee_plot.png"), dpi=FIG_DPI)
    else:
        plt.show()


def plot_scatter_fit(
    df: pd.DataFrame,
    S_fit: np.ndarray,
    equation: callable,
    params: np.ndarray,
    title: str,
    max_value: float = 1.0,
    save: bool = False,
) -> None:
    """
    Plot scatter points and fitted curve for the given DataFrame.

    Params
    ------
    - df (pd.DataFrame): DataFrame containing the data to plot.
    - S_fit (np.ndarray): Substrate concentration values for the fitted curve.
    - equation (callable): Fitting equation.
    - params (np.ndarray): Fitting parameters.
    - title (str): Title for the plot.
    - max_value (float): Maximum value for S2 to consider in the plot. Default is 1.0.
    - save (bool): Whether to save the plot as an image file. Default is False.
    """
    plt.figure(figsize=(10, 6))
    for s2, group in df[df["S2"] <= max_value].groupby("S2"):
        plt.scatter(group["S1"], group["Rate"], label=f"[S2] = {s2:.1f} mM")

    v_fit = equation(S_fit, *params)

    plt.plot(S_fit, v_fit, label="Fitted curve", color="black", linewidth=3)
    plt.xlabel("Substrate concentration [S1] (mM)")
    plt.ylabel("Rate (mM/s)")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "scatter_fit.png"), dpi=FIG_DPI)
    else:
        plt.show()
