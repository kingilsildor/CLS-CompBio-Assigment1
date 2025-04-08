import numpy as np

from script.plot_data import (
    plot_feasible_region,
    plot_lineweaver_burk,
    plot_scatter_fit,
)
from src.curve_fitting import bisubstrate_curve, michaelis_menten_curve
from src.load_data import load_data
from src.models import michaelis_menten


def figures(data):
    params, _ = michaelis_menten_curve(data["S1"], data["Rate"])
    S_fit = np.linspace(0, 1.0, 100)
    plot_scatter_fit(
        data,
        S_fit,
        michaelis_menten,
        params,
        "Michaelis-Menten Fit",
        max_value=1.0,
    )

    plot_lineweaver_burk(
        data,
        params[0],
        params[1],
        max_value=1.0,
    )


def main():
    file_path = "data/Kinetics.csv"
    data = load_data(file_path)

    params, _ = bisubstrate_curve(
        data["S1"],
        data["S2"],
        data["Rate"],
    )
    v_max = params[0]
    k_m1 = params[1]
    k_m2 = params[2]
    print(f"v_max: {v_max}, k_m1: {k_m1}, k_m2: {k_m2}")

    vmax = 12
    km = 6
    succ = 5.0
    fum = 1.0
    v6_max = 10.0
    d_target = 1.0
    plot_feasible_region(v6_max, vmax, succ, fum, km, d_target)


if __name__ == "__main__":
    main()
