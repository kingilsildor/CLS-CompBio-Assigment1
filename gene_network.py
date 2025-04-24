import matplotlib.pyplot as plt
import numpy as np


def initialize_params():
    return {
        "a_A": 1.0,
        "a_B": 0.25,
        "b_A": 0.0005,
        "b_B": 0.0005,
        "c_A": 2.0,
        "c_B": 0.5,
        "beta_A": 2.35,
        "beta_B": 2.35,
        "gamma_A": 1.0,
        "gamma_B": 1.0,
        "n_a": 3,
        "n_b": 3,
        "theta_A": 0.21,
        "theta_B": 0.21,
        "k_PA": 1.0,
        "k_PB": 1.0,
        "m_A": 2.35,
        "m_B": 2.35,
        "delta_PA": 1.0,
        "delta_PB": 1.0,
        "var_1a": 0.05,
        "var_2a": 0.05,
        "var_1b": 0.05,
        "var_2b": 0.05,
    }


def initialize_initial_conditions():
    return {
        "U_a": 0.8,
        "U_b": 0.8,
        "P_a": 0.8,
        "P_b": 0.8,
        "S_a": 0.8,
        "S_b": 0.8,
    }


def simualate_gene_network(num_simulations, N, dt, params, initial_conditions):
    U_a_total, U_b_total = (
        np.zeros((num_simulations, N)),
        np.zeros((num_simulations, N)),
    )
    P_a_total, P_b_total = (
        np.zeros((num_simulations, N)),
        np.zeros((num_simulations, N)),
    )
    S_a_total, S_b_total = (
        np.zeros((num_simulations, N)),
        np.zeros((num_simulations, N)),
    )

    for sim in range(num_simulations):
        U_a = np.zeros(N)
        U_b = np.zeros(N)
        P_a = np.zeros(N)
        P_b = np.zeros(N)
        S_a = np.zeros(N)
        S_b = np.zeros(N)

        U_a[0] = initial_conditions["U_a"]
        U_b[0] = initial_conditions["U_b"]
        P_a[0] = initial_conditions["P_a"]
        P_b[0] = initial_conditions["P_b"]
        S_a[0] = initial_conditions["S_a"]
        S_b[0] = initial_conditions["S_b"]

        for t in range(1, N):
            # Euler-Maruyama method

            dB_1A = np.random.normal(0, np.sqrt(dt))
            dB_2A = np.random.normal(0, np.sqrt(dt))
            dB_1B = np.random.normal(0, np.sqrt(dt))
            dB_2B = np.random.normal(0, np.sqrt(dt))

            betaA_crit = params["beta_A"] * (
                (P_b[t - 1] ** params["n_a"])
                / (params["theta_A"] ** params["n_a"] + P_b[t - 1] ** params["n_a"])
            )
            betaB_crit = params["beta_B"] * (
                (params["theta_B"] ** params["n_b"])
                / (params["theta_B"] ** params["n_b"] + P_a[t - 1] ** params["n_b"])
            )

            U_b[t] = (
                U_b[t - 1]
                + (params["a_B"] - betaB_crit * U_b[t - 1]) * dt
                + params["var_1b"] * dB_1B
            )
            U_a[t] = (
                U_a[t - 1]
                + (params["a_A"] - betaA_crit * U_a[t - 1]) * dt
                + params["var_1a"] * dB_1A
            )

            S_b[t] = (
                S_b[t - 1]
                + (betaB_crit * U_b[t - 1] - params["gamma_B"] * S_b[t - 1]) * dt
                + params["var_2b"] * dB_2B
            )
            S_a[t] = (
                S_a[t - 1]
                + (betaA_crit * U_a[t - 1] - params["gamma_A"] * S_a[t - 1]) * dt
                + params["var_2a"] * dB_2A
            )

            P_a[t] = (
                P_a[t - 1]
                + (params["k_PA"] * S_a[t - 1] - params["delta_PA"] * P_a[t - 1]) * dt
            )
            P_b[t] = (
                P_b[t - 1]
                + (params["k_PB"] * S_b[t - 1] - params["delta_PB"] * P_b[t - 1]) * dt
            )

        U_a_total[sim] = U_a
        U_b_total[sim] = U_b
        P_a_total[sim] = P_a
        P_b_total[sim] = P_b
        S_a_total[sim] = S_a
        S_b_total[sim] = S_b

    return U_a_total, U_b_total, P_a_total, P_b_total, S_a_total, S_b_total


def plot_results(S_a, S_b, P_a, P_b, P_b_std, S_a_std, S_b_std, T, N):
    t = np.linspace(0, T, N)
    plt.figure(figsize=(6, 5))
    plt.plot(t, S_a, label="mRNA A", color="blue")
    plt.fill_between(t, S_a - S_a_std, S_a + S_a_std, color="blue", alpha=0.2)
    plt.plot(t, S_b, label="mRNA B", color="orange")
    plt.fill_between(t, S_b - S_b_std, S_b + S_b_std, color="orange", alpha=0.2)
    plt.suptitle("mRNA A and B", fontsize=16, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Concentration (M)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig("mrna_concentrations_1b.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(P_a, P_b, label="Protein A vs Protein B", color="green")
    plt.fill_between(P_a, P_b - P_b_std, P_b + P_b_std, color="green", alpha=0.2)
    plt.plot(P_a[0], P_b[0], marker="o", color="red", label="Initial Condition")
    plt.plot(P_a[-1], P_b[-1], marker="o", color="black", label="Final Condition")
    plt.xlim(0.4, 1.5)
    plt.ylim(0, 1)
    plt.suptitle("Phase space of protein A and B", fontsize=16, fontweight="bold")
    plt.xlabel("Protein A (M)", fontsize=14)
    plt.ylabel("Protein B (M)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig("phase_space_1b.png", dpi=300)
    plt.show()
    plt.close()


def main():
    params = initialize_params()
    initial_conditions = initialize_initial_conditions()

    dt = 0.01
    T = 100
    N = int(T / dt)

    num_simulations = 1000

    U_a_total, U_b_total, P_a_total, P_b_total, S_a_total, S_b_total = (
        simualate_gene_network(num_simulations, N, dt, params, initial_conditions)
    )
    S_a_mean = np.mean(S_a_total, axis=0)
    S_b_mean = np.mean(S_b_total, axis=0)
    P_a_mean = np.mean(P_a_total, axis=0)
    P_b_mean = np.mean(P_b_total, axis=0)

    P_b_std = np.std(P_b_total, axis=0)
    S_a_std = np.std(S_a_total, axis=0)
    S_b_std = np.std(S_b_total, axis=0)
    print(S_a_mean, S_b_mean)
    print(S_a_std, S_b_std)

    plot_results(
        S_a_mean, S_b_mean, P_a_mean, P_b_mean, P_b_std, S_a_std, S_b_std, T, N
    )


if __name__ == "__main__":
    main()
