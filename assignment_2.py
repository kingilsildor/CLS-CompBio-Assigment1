import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def metabolite_model(x, y, alpha, beta):
    metabolite = alpha * x - beta * x * y
    return metabolite


def enzyme_model(x, y, gamma, delta):
    enzyme = -gamma * y + delta * x * y
    return enzyme


def ode_model(t, pop, alpha, beta, gamma, delta):
    x, y = pop
    metabolite_cst = metabolite_model(x, y, alpha, beta)
    enzyme_cst = enzyme_model(x, y, gamma, delta)
    return [metabolite_cst, enzyme_cst]


def solve_ode_system(t_max, delta_t, init_pop, alpha, beta, gamma, delta):
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


def create_grid(min_val, max_val, num_points):
    x = np.linspace(min_val, max_val, num_points)
    y = np.linspace(min_val, max_val, num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_phase_portrait(X, Y, alpha, beta, gamma, delta):
    U = metabolite_model(X, Y, alpha, beta)
    V = enzyme_model(X, Y, gamma, delta)

    plt.streamplot(X, Y, U, V, color="gray", linewidth=1, arrowsize=1.5)
    plt.xlabel("Metabolite Concentration ($x$)")
    plt.ylabel("Enzyme Concentration ($y$)")
    plt.plot(gamma / delta, alpha / beta, "ro", markersize=8, label="Equilibrium")
    # plt.axhline(0, color="r", linestyle="--", label="$dx/dt=0$")
    # plt.axvline(0, color="b", linestyle="--", label="$dy/dt=0$")

    plt.title("Phase Portrait of Metabolite-Enzyme System")
    plt.legend()
    plt.show()


def plot_ode_solution(sol):
    x = sol.y[0]
    y = sol.y[1]
    t = sol.t
    plt.plot(t, x, label="Metabolite")
    plt.plot(t, y, label="Enzyme")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Metabolite and Enzyme Concentrations Over Time")
    plt.legend()
    plt.show()


def main():
    # sol = solve_ode_system(
    #     t_max=10,
    #     delta_t=0.1,
    #     init_pop=(1, 0.5),
    #     alpha=2,
    #     beta=1.1,
    #     gamma=1,
    #     delta=0.9,
    # )
    # plot_ode_solution(sol)

    X, Y = create_grid(0, 5, 20)
    plot_phase_portrait(X, Y, alpha=2, beta=1.1, gamma=1, delta=0.9)


if __name__ == "__main__":
    main()
