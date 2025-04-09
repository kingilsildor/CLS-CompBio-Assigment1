import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

FIG_DPI = 300


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def chi_squared(y_true, y_pred):
    return np.sum(((y_true - y_pred) ** 2) / y_true)


def sequential_model(S, V_max, K_m1, K_m2):
    S1, S2 = S
    v = (V_max * S1 * S2) / (K_m1 * K_m2 + K_m2 * S1 + K_m1 * S2 + S1 * S2)
    return v


def ping_pong_model(S, V_max, K_m1, K_m2):
    S1, S2 = S
    v = (V_max * S1 * S2) / (K_m1 * S2 + K_m2 * S1 + S1 * S2)
    return v


def fit_sequential_bisubstrate(data, model):
    S = data[["S1", "S2"]].values.T
    rates = data["Rate"].values

    params, _ = curve_fit(model, S, rates)

    predicted_rates = model(S, *params)
    r2_score = r_squared(rates, predicted_rates)
    chi2_score = chi_squared(rates, predicted_rates)

    return (r2_score, chi2_score, params)


def compare_output(tuple_output, titles):
    for output, title in zip(tuple_output, titles):
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
    model,
    data,
    s2_values,
    simulate_amount=100,
):
    params, _ = curve_fit(model, (data["S1"], data["S2"]), data["Rate"])

    s1_range = data["S1"].unique()
    s1_combined = np.tile(s1_range, len(s2_values))
    s2_combined = np.repeat(s2_values, len(s1_range))

    rates = model((s1_combined, s2_combined), *params)
    simulated_data = pd.DataFrame({"S1": s1_combined, "S2": s2_combined, "Rate": rates})

    return simulated_data


def plot_eadie_hofstee(df, save=False):
    constants = {}
    plt.figure(figsize=(10, 6))

    s2_values = df["S2"].unique()
    for s2 in s2_values:
        df_substrate = df[df["S2"] == s2]

        x = df_substrate["Rate"] / df_substrate["S1"]
        y = df_substrate["Rate"]

        K_m1, v_max = np.polyfit(x, y, 1)
        constants[s2] = (K_m1, v_max)

        plt.scatter(x, y, label=f"[S2] = {s2} mM")
        plt.plot(
            x,
            K_m1 * x + v_max,
            "--",
            label=f"$K_{{m1}} = {K_m1:.2f}$ mM/s, $v_{{max}} = {v_max:.2f}$ mM/s",
        )

    plt.xlabel(r"v / [S1] (mH$^{{-1}}s^{{-1}}$)")
    plt.ylabel(r"v (mM/s)")
    plt.title("Eadie-Hofstee Plot")
    plt.legend(loc="upper right")

    plt.tight_layout()
    if save:
        plt.savefig("results/eadie_hofstee_plot.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

    constants = pd.DataFrame.from_dict(
        constants, orient="index", columns=["K_m1", "V_max"]
    )
    constants.index.name = "S2"
    constants.reset_index(inplace=True)
    return constants


def plot_lineweaver_burk(df, save=False):
    plt.figure(figsize=(10, 6))

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


def calculate_constants(S2, v, v_max, K_m1):
    S1 = 1.0
    K_m2 = (v_max * S1 * S2 - v * K_m1 * S2 - v * S1 * S2) / (v * S1)
    return K_m2


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

    df_simulated_data.to_csv("data/simulated_data.csv", index=False)

    constants = plot_eadie_hofstee(df_simulated_data, save=True)
    plot_lineweaver_burk(df_simulated_data, save=True)

    print(df_simulated_data.head())

    for S2 in constants["S2"]:
        v_max = constants[constants["S2"] == S2]["V_max"].values[0]
        K_m1 = constants[constants["S2"] == S2]["K_m1"].values[0]
        # K_m2 = calculate_constants(S2, v, v_max, K_m1)
        # print(f"S2: {S2} mM")
        # print(f"v: {v:.4f} mM/s")
        # print(f"v_max: {v_max:.4f} mM/s")
        # print(f"K_m1: {K_m1:.4f} mM")
        # print(f"K_m2: {K_m2:.4f} mM\n")


if __name__ == "__main__":
    main()
