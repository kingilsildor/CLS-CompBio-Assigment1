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
    - save (bool): If True, save the plot to a file.
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
    - save (bool): If True, save the plot to a file.
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
        plt.savefig("results/protein_phase_plane.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()

def plot_protein_dynamics_solution(sol: OdeResult, save=False) -> None:
    """
    Plot the solution of the ODE system.

    Params
    ----------
    - sol (OdeResult): Solution object from solve_ivp.
    - save (bool): If True, save the plot to a file.
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
    plt.title("Metabolite and Enzyme Concentrations Over Time")
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

###########################################

def main(SAVING: bool = False) -> None:
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
        delta_t=0.1,
        init_pop=(list(init_pop.values())),
        params=(list(params_dict.values()))
    )
    plot_mRNA_concentrations_solution(sol, save=SAVING)
    plot_protein_dynamics_solution(sol, save=SAVING)    


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
        delta_t=0.1,
        init_pop=(list(init_pop.values())),
        params=(list(params_dict.values()))
    )
    plot_ode_solution(sol, save=SAVING)

    X, Y = create_grid(-1, 3, 20)
    X, Y = create_grid(-0.5, 2.5, 20)
    plot_phase_portrait(X, Y, **params_dict, save=SAVING)


if __name__ == "__main__":
    main(True)
