import numpy as np

from script.plot_data import plot_eadie_hofstee, plot_scatter_fit
from src.curve_fitting import michaelis_menten_curve
from src.load_data import load_data
from src.models import bisubstrate_rate, michaelis_menten


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

    plot_eadie_hofstee(data, params[0], params[1], max_value=1.0, save=False)


def eadie_hofstee(data):
    params, _ = michaelis_menten_curve(data["S1"], data["Rate"])
    v_max, k_m = params
    v = bisubstrate_rate(
        (data["S1"], data["S2"]),
        v_max,
        k_m,
        k_m,
    )
    print(v)


def main():
    file_path = "data/Kinetics.csv"
    data = load_data(file_path)
    figures(data)


if __name__ == "__main__":
    main()
