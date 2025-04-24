import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import os

######## Setup Plotting Sytle ############
FIG_DPI = 400
STRD_FIG_SIZE = (10, 6)
SQ_FIG_SIZE = (9, 9)

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

##########################################
def simualate_gene_network(num_simulations: int, N: int, dt: float, params: dict, initial_conditions: dict) -> tuple:
    """
    Simulate the gene network using the Euler-Maruyama method.

    Params
    ----------
    - num_simulations (int): Number of simulations to run.
    - N (int): Number of time steps.
    - dt (float): Time step size.
    - params (dict): Dictionary of parameters for the model.
    - initial_conditions (dict): Dictionary of initial conditions for the model.

    Returns
    ----------
    - tuple: Tuple containing the simulated concentrations of U_a, U_b, P_a, P_b, S_a, and S_b.
    """
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

def hill_function(x: float, theta: float, n: int) -> float:
    """
    Hill function for gene regulation.

    Params
    ----------
    - x (float): Concentration of the molecule.
    - theta (float): Threshold concentration.
    - n (int): Hill coefficient.

    Returns
    ----------
    - float: Value of the Hill function.
    """
    return (x ** n) / (theta ** n + x ** n)

def gene_regulation_ode(t: float, pop: np.ndarray, params: list) -> list:
    """
    ODE system for gene regulation.

    Params
    ----------
    - t (float): Time variable.
    - pop (ndarray): List containing mRNA and protein concentrations.
    - params (list): List of parameters for the model.

    Returns
    ----------
    - list: List containing the rate of change of mRNA and protein concentrations.
    """
    m_A, m_B, gamma_A, gamma_B, k_PA, k_PB, theta_A, theta_B, n_A, n_B, delta_PA, delta_PB = params
    mRNA_A, mRNA_B, Protein_A, Protein_B = pop

    hill_A = hill_function(Protein_A, theta_A, n_A)
    hill_B = hill_function(Protein_B, theta_B, n_B)

    # Protein A inhibits gene B transcription
    dmRNA_A_dt = m_A * hill_B - gamma_A * mRNA_A
    inhibition_A = theta_A ** n_A / (theta_A ** n_A + Protein_A ** n_A)
    # Protein B activates gene A transcription
    dmRNA_B_dt = m_B * inhibition_A - gamma_B * mRNA_B
    dProtein_A_dt = k_PA * mRNA_A - delta_PA * Protein_A
    dProtein_B_dt = k_PB * mRNA_B - delta_PB * Protein_B

    return [dmRNA_A_dt, dmRNA_B_dt, dProtein_A_dt, dProtein_B_dt]


def metabolite_ode(t: float, pop: np.ndarray, params: list) -> list:
    """
    Metabolite ODE model function.

    Params
    ----------
    - t (float): Time variable.
    - pop (ndarray): List containing metabolite and enzyme concentrations.
    - params (list): List of parameters for the model.

    Returns
    ----------
    - list: List containing the rate of change of metabolite and enzyme concentrations.
    """
    alpha, beta, gamma, delta = params
    x, y = pop

    metabolite_cst = alpha * x - beta * x * y
    enzyme_cst = -gamma * y + delta * x * y
    return [metabolite_cst, enzyme_cst]


def solve_ode_system(model: callable, t_max: float, delta_t: float, init_pop: list, params: list) -> OdeResult:
    """
    Solve the ODE system using the Runge-Kutta method.

    Params
    ----------
    - model (callable): ODE model function.
    - t_max (float): Maximum time for the simulation.
    - delta_t (float): Time step for the simulation.
    - init_pop (list): Initial population of metabolite and enzyme concentrations.
    - params (list): List of parameters for the model.

    Returns
    ----------
    - sol (OdeResult): Solution object containing the results of the ODE integration.
    """
    t_span = np.linspace(0, t_max, int(t_max / delta_t) + 1)

    sol = solve_ivp(
        model,
        [0, t_max],
        init_pop,
        args=(params,),
        t_eval=t_span,
        rtol=1e-6,
        atol=1e-9
    )
    return sol


def create_grid(min_val: float, max_val: float, num_points: int) -> tuple:
    """
    Create a grid of points for the phase portrait.

    Params
    ----------
    - min_val (float): Minimum value for the grid.
    - max_val (float): Maximum value for the grid.
    - num_points (int): Number of points in each dimension.

    Returns
    ----------
    - X (ndarray): 2D array of x-coordinates.
    - Y (ndarray): 2D array of y-coordinates.
    """
    x = np.linspace(min_val, max_val, num_points)
    y = np.linspace(min_val, max_val, num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y

######## Plotting Functions ############
def plot_phase_portrait(X: np.ndarray, Y: np.ndarray, alpha: float, beta: float, gamma: float, delta: float, save=False) -> None:
    """
    Plot the phase portrait of the metabolite-enzyme system.

    Params
    ----------
    - X (ndarray): 2D array of x-coordinates.
    - Y (ndarray): 2D array of y-coordinates.
    - alpha (float): Parameter for metabolite production.
    - beta (float): Parameter for metabolite consumption.
    - gamma (float): Parameter for enzyme production.
    - delta (float): Parameter for enzyme consumption.
    - save (bool): If True, save the plot to a file. Default is False.
    """
    plt.figure(figsize=STRD_FIG_SIZE)

    U = alpha * X - beta * X * Y
    V = -gamma * Y + delta * X * Y

    plt.streamplot(X, Y, U, V, color="gray", linewidth=1, arrowsize=0.5)
    plt.xlabel("Metabolite (x) Concentration (M)")
    plt.ylabel("Enzyme (y) Concentration (M)")
    plt.plot(gamma / delta, alpha / beta, "ro", markersize=8, label=r"Equilibrium = $\left(\frac{\gamma}{\delta}, \frac{\alpha}{\beta}\right)$")
    print(f"Equilibrium point: ({gamma / delta:.2f}, {alpha / beta:.2f})")
    plt.axhline(0, color="r", linestyle="--", label="$dx/dt=0$")
    plt.axvline(0, color="b", linestyle="--", label="$dy/dt=0$")

    plt.title("Phase Portrait of Metabolite-Enzyme System")
    plt.legend()

    plt.tight_layout()
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig("results/phase_plot.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_ode_solution(sol: OdeResult, save=False) -> None:
    """
    Plot the solution of the ODE system.

    Params
    ----------
    - sol (OdeResult): Solution object from solve_ivp.
    - save (bool): If True, save the plot to a file. Default is False.
    """
    plt.figure(figsize=STRD_FIG_SIZE)

    x = sol.y[0]
    y = sol.y[1]
    t = sol.t

    plt.plot(t, x, label="Metabolite")
    plt.plot(t, y, label="Enzyme")
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.title("Metabolite and Enzyme Concentrations Over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig("results/ode_solution_plot.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

def plot_mRNA_concentrations_solution(sol: OdeResult, save=False) -> None:
    """
    Plot the solution of the ODE system.

    Params
    ----------
    - sol (OdeResult): Solution object from solve_ivp.
    - save (bool): If True, save the plot to a file. Default is False.
    """
    plt.figure(figsize=STRD_FIG_SIZE)

    mRNA_A = sol.y[0]
    mRNA_B = sol.y[1]
    t = sol.t

    plt.plot(t, mRNA_A, 'b-', label='mRNA A')
    plt.plot(t, mRNA_B, 'r-', label='mRNA B')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.title('Time Evolution of mRNA A and mRNA B Concentrations')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig("results/mrna_time_evolution.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

def plot_protein_dynamics_solution(sol: OdeResult, save=False) -> None:
    """
    Plot the solution of the ODE system.

    Params
    ----------
    - sol (OdeResult): Solution object from solve_ivp.
    - save (bool): If True, save the plot to a file. Default is False.
    """
    plt.figure(figsize=SQ_FIG_SIZE)

    Protein_A = sol.y[2]
    Protein_B = sol.y[3]
    t = sol.t

    plt.plot(Protein_A, Protein_B, 'g-')
    plt.plot(Protein_A[0], Protein_B[0], 'go', label='Start')  # Start point
    plt.plot(Protein_A[-1], Protein_B[-1], 'ro', label='End')  # End point
  
    plt.xlabel('Protein A Concentration (M)')
    plt.ylabel('Protein B Concentration (M)')
    plt.title('Phase-Plane Plot of Protein A vs Protein B')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig("results/protein_phase_plane.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

def plot_stochastic_mrna(S_a: np.ndarray, S_b: np.ndarray, S_a_std: np.ndarray, S_b_std: np.ndarray, T: float, N: int, save=False) -> None:
    """
    Plot the stochastic mRNA concentrations.
    Params
    ----------
    - S_a (ndarray): mRNA A concentrations.
    - S_b (ndarray): mRNA B concentrations.
    - S_a_std (ndarray): Standard deviation of mRNA A concentrations.
    - S_b_std (ndarray): Standard deviation of mRNA B concentrations.
    - T (float): Total time.
    - N (int): Number of time points.
    - save (bool): If True, save the plot to a file. Default is False.
    """
    t = np.linspace(0, T, N)

    plt.figure(figsize=STRD_FIG_SIZE)

    plt.plot(t, S_a, label="mRNA A", color="blue")
    plt.fill_between(t, S_a - S_a_std, S_a + S_a_std, color="blue", alpha=0.2)
    plt.plot(t, S_b, label="mRNA B", color="orange")
    plt.fill_between(t, S_b - S_b_std, S_b + S_b_std, color="orange", alpha=0.2)

    plt.title("mRNA A and B")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (M)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig("results/mrna_concentrations_1b.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

def plot_stochastic_protein(P_a: np.ndarray, P_b: np.ndarray, P_b_std: np.ndarray, save=False) -> None:
    plt.figure(figsize=STRD_FIG_SIZE)

    plt.plot(P_a, P_b, label="Protein A vs Protein B", color="green")
    plt.fill_between(P_a, P_b - P_b_std, P_b + P_b_std, color="green", alpha=0.2)
    plt.plot(P_a[0], P_b[0], marker="o", color="red", label="Initial Condition")
    plt.plot(P_a[-1], P_b[-1], marker="o", color="black", label="Final Condition")
    
    plt.xlim(0.4, 1.5)
    plt.ylim(0, 1)

    plt.title("Phase space of protein A and B")
    plt.xlabel("Protein A (M)")
    plt.ylabel("Protein B (M)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig("results/phase_space_1b.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

###########################################
def main(SAVING: bool = False) -> None:
    dt = 0.01

    ################### Question 1 ###################
    params_dict = {
        "m_A": 2.35,
        "m_B": 2.35,
        "gamma_A": 1,
        "gamma_B": 1,
        "k_PA": 1,
        "k_PB": 1,
        "theta_A": 0.21,
        "theta_B": 0.21,
        "n_A": 3,
        "n_B": 3,
        "delta_PA": 1,
        "delta_PB": 1
    }
    init_pop = {
        "mRNA_A": 0.8,
        "mRNA_B": 0.8,
        "Protein_A": 0.8,
        "Protein_B": 0.8
    }
    
    sol = solve_ode_system(
        model=gene_regulation_ode,
        t_max=100,
        delta_t=dt,
        init_pop=(list(init_pop.values())),
        params=(list(params_dict.values()))
    )
    plot_mRNA_concentrations_solution(sol, save=SAVING)
    plot_protein_dynamics_solution(sol, save=SAVING)

    ################### Question 1 ###################
    params_dict = {
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
    init_pop = {
        "U_a": 0.8,
        "U_b": 0.8,
        "P_a": 0.8,
        "P_b": 0.8,
        "S_a": 0.8,
        "S_b": 0.8,
    }

    T = 100
    N = int(T / dt)

    num_simulations = 1000

    _, _, P_a_total, P_b_total, S_a_total, S_b_total = (
        simualate_gene_network(num_simulations, N, dt, params_dict, init_pop)
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

    plot_stochastic_mrna(S_a_mean, S_b_mean, S_a_std, S_b_std, T, N, save=SAVING)
    plot_stochastic_protein(P_a_mean, P_b_mean, P_b_std, save=SAVING)
    
    ################### Question 2 ###################
    params_dict = {
        "alpha": 2,
        "beta": 1.1,
        "gamma": 1,
        "delta": 0.9,
    }
    init_pop = {
        "metabolite": 1,
        "enzyme": 0.5
    }

    sol = solve_ode_system(
        model=metabolite_ode,
        t_max=10,
        delta_t=dt,
        init_pop=(list(init_pop.values())),
        params=(list(params_dict.values()))
    )
    plot_ode_solution(sol, save=SAVING)

    X, Y = create_grid(-1, 3, 20)
    X, Y = create_grid(-0.5, 2.5, 20)
    plot_phase_portrait(X, Y, **params_dict, save=SAVING)


if __name__ == "__main__":
    main(True)
