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


def plot_feasible_region(v6_max, vmax, succ, fum, km, d_target):
    """
    Plot the feasible region for the given parameters.

    Params
    ------
    - v6_max (float): Maximum reaction rate for the sixth reaction.
    - vmax (float): Maximum reaction rate for the first reaction.
    - succ (float): Concentration of succinate.
    - fum (float): Concentration of fumarate.
    - km (float): Michaelis constant.
    """

    v6_MM = vmax * (succ / (km + succ)) - vmax * (fum / (km + fum))
    v6_effective = min(v6_MM, v6_max)

    v1_values = np.linspace(0, v6_effective, 100)
    v6_diagonal = v1_values.copy()

    v6_horizontal = v6_effective * np.ones_like(v1_values)

    d_line = d_target + v1_values

    plt.figure(figsize=(8, 6))
    plt.plot(v1_values, v6_diagonal, "r--", linewidth=2, label=r"$v_6 = v_1$")
    plt.plot(
        v1_values,
        v6_horizontal,
        "b-",
        linewidth=2,
        label=r"$v_6 \leq v_{6,\max}$",
    )
    plt.fill_between(
        v1_values,
        v6_diagonal,
        v6_horizontal,
        color="gray",
        alpha=0.3,
        label=r"Feasible Region: $v_1 < v_6 \leq$ $v_{6,max}$",
    )
    plt.plot(
        v1_values, d_line, "m-", linewidth=2, label=r"$v_6 = v_1 + D_{\rm target}$"
    )

    plt.xlabel(r"$v_1$", fontsize=14)
    plt.ylabel(r"$v_6$", fontsize=14)
    plt.title(
        "Solution Space in (v₁, v₆)",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.xlim(0, v6_effective + 0.5)
    plt.ylim(0, v6_effective + 0.5)
    plt.grid(True)
    plt.show()
