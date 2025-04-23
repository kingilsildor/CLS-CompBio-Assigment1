import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

######## Setup Plotting Sytle ############
FIG_DPI = 400
FIG_SIZE = (10, 6)

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
##########################################


def metabolite_model(x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Metabolite model function.

    Params
    ----------
    - x (ndarray): Metabolite concentration.
    - y (ndarray): Enzyme concentration.
    - alpha (float): Parameter for metabolite production.
    - beta (float): Parameter for metabolite consumption.

    Returns
    ----------
    - metabolite (ndarray): Rate of change of metabolite concentration.
    """
    metabolite = alpha * x - beta * x * y
    return metabolite


def enzyme_model(x: np.ndarray, y: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    """
    Enzyme model function.

    Params
    ----------
    - x (ndarray): Metabolite concentration.
    - y (ndarray): Enzyme concentration.
    - gamma (float): Parameter for enzyme production.
    - delta (float): Parameter for enzyme consumption.

    Returns
    ----------
    - enzyme (ndarray): Rate of change of enzyme concentration.
    """
    enzyme = -gamma * y + delta * x * y
    return enzyme


def ode_model(t: float, pop: tuple, alpha: float, beta: float, gamma: float, delta: float) -> list:
    """
    ODE model function.

    Params
    ----------
    - t (float): Time variable.
    - pop (tuple): Tuple containing metabolite and enzyme concentrations.
    - alpha (float): Parameter for metabolite production.
    - beta (float): Parameter for metabolite consumption.
    - gamma (float): Parameter for enzyme production.
    - delta (float): Parameter for enzyme consumption.

    Returns
    ----------
    - list: List containing the rate of change of metabolite and enzyme concentrations.
    """
    x, y = pop
    metabolite_cst = metabolite_model(x, y, alpha, beta)
    enzyme_cst = enzyme_model(x, y, gamma, delta)
    return [metabolite_cst, enzyme_cst]


def solve_ode_system(t_max: float, delta_t: float, init_pop: tuple, alpha: float, beta: float, gamma: float, delta: float) -> OdeResult:
    """
    Solve the ODE system using the Runge-Kutta method.

    Params
    ----------
    - t_max (float): Maximum time for the simulation.
    - delta_t (float): Time step for the simulation.
    - init_pop (tuple): Initial population of metabolite and enzyme concentrations.
    - alpha (float): Parameter for metabolite production.
    - beta (float): Parameter for metabolite consumption.
    - gamma (float): Parameter for enzyme production.
    - delta (float): Parameter for enzyme consumption.

    Returns
    ----------
    - sol (OdeResult): Solution object containing the results of the ODE integration.
    """
    x0, y0 = init_pop
    t_span = np.linspace(0, t_max, int(t_max / delta_t) + 1)

    sol = solve_ivp(
        ode_model,
        [0, t_max],
        [x0, y0],
        args=(alpha, beta, gamma, delta),
        method="RK45",
        t_eval=t_span,
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
    plt.figure(figsize=FIG_SIZE)

    U = metabolite_model(X, Y, alpha, beta)
    V = enzyme_model(X, Y, gamma, delta)

    plt.streamplot(X, Y, U, V, color="gray", linewidth=1, arrowsize=0.5)
    plt.xlabel("Metabolite Concentration ($x$)")
    plt.ylabel("Enzyme Concentration ($y$)")
    plt.plot(gamma / delta, alpha / beta, "ro", markersize=8, label=r"Equilibrium = $\left(\frac{\gamma}{\delta}, \frac{\alpha}{\beta}\right)$")
    print(f"Equilibrium point: ({gamma / delta:.2f}, {alpha / beta:.2f})")
    plt.axhline(0, color="r", linestyle="--", label="$dx/dt=0$")
    plt.axvline(0, color="b", linestyle="--", label="$dy/dt=0$")

    plt.title("Phase Portrait of Metabolite-Enzyme System")
    plt.legend()

    plt.tight_layout()
    if save:
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
    plt.figure(figsize=FIG_SIZE)

    x = sol.y[0]
    y = sol.y[1]
    t = sol.t

    plt.plot(t, x, label="Metabolite")
    plt.plot(t, y, label="Enzyme")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Metabolite and Enzyme Concentrations Over Time")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig("results/ode_solution_plot.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def main():
    ################### Question 1 ###################

    ################### Question 2 ###################
    sol = solve_ode_system(
        t_max=10,
        delta_t=0.1,
        init_pop=(1, 0.5),
        alpha=2,
        beta=1.1,
        gamma=1,
        delta=0.9,
    )
    plot_ode_solution(sol, save=True)

    X, Y = create_grid(-0.5, 2.5, 20)
    plot_phase_portrait(X, Y, alpha=2, beta=1.1, gamma=1, delta=0.9, save=True)


if __name__ == "__main__":
    main()
