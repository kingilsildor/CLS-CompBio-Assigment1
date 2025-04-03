import os

import matplotlib.pyplot as plt
import pandas as pd

from src.config import *


def plot_lineweaver_burk(df: pd.DataFrame, save: bool = False) -> None:
    plt.figure(figsize=(10, 6))
    for s2, group in df[df["S2"] <= 1.0].groupby("S2"):
        inv_S1 = 1 / group["S1"]
        inv_Rate = 1 / group["Rate"]
        plt.plot(inv_S1, inv_Rate, "o-", label=f"1/[S2] = {1 / s2:.1f} mM⁻¹")

    plt.xlabel("1/[S1] (mM⁻¹)")
    plt.ylabel("1/Rate (s/mM)")
    plt.title("Lineweaver-Burk Plot for different values of [S2]")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "lineweaver_burk_plot.png"), dpi=FIG_DPI)
    else:
        plt.show()
