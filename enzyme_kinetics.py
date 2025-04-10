import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

######## Setup Plotting Sytle ############
FIG_DPI = 400
FIG_SIZE = (10, 6)

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
##########################################


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R-squared value for the given true and predicted values.

    Params
    ------
    - y_true (np.ndarray): True values.
    - y_pred (np.ndarray): Predicted values.

    Returns
    -------
    - float: R-squared value.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def chi_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the chi-squared value for the given true and predicted values.

    Params
    ------
    - y_true (np.ndarray): True values.
    - y_pred (np.ndarray): Predicted values.

    Returns
    -------
    - float: Chi-squared value.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    return np.sum(((y_true - y_pred) ** 2) / y_true)


def sequential_model(S: tuple, V_max: float, K_m1: float, K_m2: float) -> float:
    """
    Sequential bisubstrate model for enzyme kinetics.

    Params
    ------
    - S (tuple): Substrate concentrations (S1, S2).
    - V_max (float): Maximum reaction rate.
    - K_m1 (float): Michaelis constant for substrate 1.
    - K_m2 (float): Michaelis constant for substrate 2.

    Returns
    -------
    - float: Reaction rate.
    """
    S1, S2 = S
    v = (V_max * S1 * S2) / (K_m1 * K_m2 + K_m2 * S1 + K_m1 * S2 + S1 * S2)
    return v


def ping_pong_model(S: tuple, V_max: float, K_m1: float, K_m2: float) -> float:
    """
    Ping Pong bisubstrate model for enzyme kinetics.

    Params
    ------
    - S (tuple): Substrate concentrations (S1, S2).
    - V_max (float): Maximum reaction rate.
    - K_m1 (float): Michaelis constant for substrate 1.
    - K_m2 (float): Michaelis constant for substrate 2.

    Returns
    -------
    - float: Reaction rate.
    """
    S1, S2 = S
    v = (V_max * S1 * S2) / (K_m1 * S2 + K_m2 * S1 + S1 * S2)
    return v


def calculate_k2(
    S1: float,
    S2: float,
    v: float,
    V_max: float,
    K_m1: float,
) -> float:
    """
    Calculate K_m2 using the ping-pong model.

    Params
    ------
    - S1 (float): Concentration of substrate 1.
    - S2 (float): Concentration of substrate 2.
    - v (float): Reaction rate.
    - V_max (float): Maximum reaction rate.
    - K_m1 (float): Michaelis constant for substrate 1.

    Returns
    -------
    - float: K_m2 value.
    """
    K_m2 = (V_max * S1 * S2 - v * K_m1 * S2 - v * S1 * S2) / (v * S1)
    return K_m2


def fit_sequential_bisubstrate(data: pd.DataFrame, model: callable) -> tuple:
    """
    Fit the sequential bisubstrate model to the data.

    Params
    ------
    - data (pd.DataFrame): Data containing substrate concentrations and rates.
    - model (function): The model function to fit.

    Returns
    -------
    - tuple: R-squared value, chi-squared value, and fitted parameters.
    """
    S = data[["S1", "S2"]].values.T
    rates = data["Rate"].values

    params, _ = curve_fit(model, S, rates)

    predicted_rates = model(S, *params)
    r2_score = r_squared(rates, predicted_rates)
    chi2_score = chi_squared(rates, predicted_rates)

    return (r2_score, chi2_score, params)


def compare_output(tuple_output: tuple, titles: tuple) -> str:
    """
    Compare the outputs of two models and print the results.
    Save the results to a file and return the model with the better fit.

    Params
    ------
    - tuple_output (tuple): Tuple containing the outputs of the two models.
    - titles (tuple): Tuple containing the titles of the two models.

    Returns
    -------
    - str: The model with the better fit.
    """
    assert len(tuple_output) == 2, "tuple_output must contain two elements."

    with open("data/model_results.txt", "w") as file:
        for output, title in zip(tuple_output, titles):
            file.write(f"{title} Model Results:\n")
            file.write(f"R-squared: {output[0]:.4f}\n")
            file.write(f"Chi-squared: {output[1]:.4f}\n")
            file.write(f"Parameters: {output[2]}\n\n")

            print(f"{title} Model Results:")
            print(f"R-squared: {output[0]:.4f}")
            print(f"Chi-squared: {output[1]:.4f}")
            print(f"Parameters: {output[2]}\n")

    seq_output, ping_output = tuple_output
    if seq_output[0] > ping_output[0] and seq_output[1] < ping_output[1]:
        return "seq"
    elif ping_output[0] > seq_output[0] and ping_output[1] < seq_output[1]:
        return "ping"
    else:
        return "Both models are equally good or bad."


def simulate_data(
    model: callable,
    data: pd.DataFrame,
    s2_values: np.ndarray,
    s1_original: bool = True,
    simulate_amount: int = 100,
):
    """
    Simulate data using the fitted model.

    Params
    ------
    - model (function): The model function to use for simulation.
    - data (pd.DataFrame): Data containing substrate concentrations and rates.
    - s2_values (np.ndarray): List of S2 values to simulate.
    - s1_original (bool): Whether to use the original S1 values or a range. Default is True.
    - simulate_amount (int): Number of points to simulate. Default is 100.

    Returns
    -------
    - pd.DataFrame: Simulated data.
    """
    params, _ = curve_fit(model, (data["S1"], data["S2"]), data["Rate"])

    if s1_original:
        s1_range = data["S1"].unique()
    else:
        s1_range = np.linspace(0.1, 10, simulate_amount)

    s1_combined = np.tile(s1_range, len(s2_values))
    s2_combined = np.repeat(s2_values, len(s1_range))

    v = model((s1_combined, s2_combined), *params)

    simulated_data = pd.DataFrame(
        {
            "S1": s1_combined,
            "S2": s2_combined,
            "Rate": v,
        }
    )

    return simulated_data


def plot_eadie_hofstee(df: pd.DataFrame, save=False) -> None:
    """
    Plot the Eadie-Hofstee plot for the given data.

    Params
    ------
    - df (pd.DataFrame): Data containing substrate concentrations and rates.
    - save (bool): Whether to save the plot. Default is False.
    """
    plt.figure(figsize=FIG_SIZE)
    km2_results = []

    s2_values = df["S2"].unique()
    for s2 in s2_values:
        df_substrate = df[df["S2"] == s2]
        S1 = df_substrate["S1"].values
        v = df_substrate["Rate"].values
        v_S1 = v / S1

        K_m1, V_max = np.polyfit(v_S1, v, 1)

        S1_fixed = 1.0
        v_fixed = ping_pong_model((S1_fixed, s2), V_max, K_m1, 0.1)

        K_m2 = calculate_k2(
            S1_fixed,
            s2,
            v_fixed,
            V_max,
            K_m1,
        )
        km2_results.append(
            {
                "S1": S1_fixed,
                "S2": s2,
                "v": v_fixed,
                "V_max": V_max,
                "K_m1": -K_m1,
                "K_m2": K_m2,
            }
        )

        plt.scatter(v_S1, v, label=f"[S2] = {s2} mM")
        plt.plot(
            v_S1,
            K_m1 * v_S1 + V_max,
            "--",
            label=f"$-K_{{m1}} = {-K_m1:.2f}$, $V_{{max}} = {V_max:.2f}$",
        )

    plt.xlabel(r"v / [S1] (mM$s^{{-1}}$)")
    plt.ylabel(r"v (mM/s)")
    plt.title("Eadie-Hofstee Plot")
    plt.legend(loc="upper right")

    df = pd.DataFrame(km2_results)
    print(df)
    df.to_csv("data/km2_results.csv", index=False)

    plt.tight_layout()
    if save:
        plt.savefig("results/eadie_hofstee_plot.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_lineweaver_burk(df: pd.DataFrame, save=False) -> None:
    """
    Plot the Lineweaver-Burk plot for the given data.

    Params
    ------
    - df (pd.DataFrame): Data containing substrate concentrations and rates.
    - save (bool): Whether to save the plot. Default is False.
    """
    plt.figure(figsize=FIG_SIZE)

    for s2, group in df.groupby("S2"):
        inv_S1 = 1 / group["S1"]
        inv_Rate = 1 / group["Rate"]
        plt.plot(inv_S1, inv_Rate, "o-", label=f"[S2] = {s2} mM")

    plt.xlabel(r"1/[S1] (mM$^{{-1}})$")
    plt.ylabel(r"1/v (s/mM)")
    plt.title("Lineweaver-Burk Plot")
    plt.legend(loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig("results/lineweaver_burk_plot.png", dpi=FIG_DPI)
    else:
        plt.show()


def main():
    data = pd.read_csv("data/Kinetics.csv")

    seq_output = fit_sequential_bisubstrate(data, sequential_model)
    ping_output = fit_sequential_bisubstrate(data, ping_pong_model)
    model_str = compare_output((seq_output, ping_output), ("Sequential", "Ping Pong"))

    s2_range = [1.5, 2.5, 5.0]
    sim_amount = 100

    if model_str == "seq":
        df_simulated_data = simulate_data(
            sequential_model, data, s2_range, simulate_amount=sim_amount
        )
    elif model_str == "ping":
        df_simulated_data = simulate_data(
            ping_pong_model, data, s2_range, simulate_amount=sim_amount
        )
    else:
        raise ValueError("Both models are equally good or bad.")

    plot_eadie_hofstee(df_simulated_data, save=True)
    plot_lineweaver_burk(df_simulated_data, save=True)


if __name__ == "__main__":
    main()
