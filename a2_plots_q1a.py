import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')  # To avoid PyCharm bug
import matplotlib.pyplot as plt


# Define the ODE system for Route I (transcription factor regulation)
def gene_regulation_ode(t, y, params):
    mRNA_A, mRNA_B, Protein_A, Protein_B = y

    # Extract parameters
    m_A, m_B, gamma_A, gamma_B, k_PA, k_PB, theta_A, theta_B, n_A, n_B, delta_PA, delta_PB = params

    # Hill functions for regulation
    hill_A = (Protein_A ** n_A) / (theta_A ** n_A + Protein_A ** n_A)
    hill_B = (Protein_B ** n_B) / (theta_B ** n_B + Protein_B ** n_B)

    # Differential equations
    # Protein A inhibits gene B transcription
    # Protein B activates gene A transcription
    dmRNA_A_dt = m_A * hill_B - gamma_A * mRNA_A
    inhibition_A = theta_A ** n_A / (theta_A ** n_A + Protein_A ** n_A)
    dmRNA_B_dt = m_B * inhibition_A - gamma_B * mRNA_B
    dProtein_A_dt = k_PA * mRNA_A - delta_PA * Protein_A
    dProtein_B_dt = k_PB * mRNA_B - delta_PB * Protein_B

    return [dmRNA_A_dt, dmRNA_B_dt, dProtein_A_dt, dProtein_B_dt]


# Parameters from Table 5
params = [
    2.35,  # m_A (max transcription rate of Gene A)
    2.35,  # m_B (max transcription rate of Gene B)
    1.0,  # gamma_A (mRNA A degradation rate)
    1.0,  # gamma_B (mRNA B degradation rate)
    1.0,  # k_PA (translation rate of Protein A)
    1.0,  # k_PB (translation rate of Protein B)
    0.21,  # theta_A (expression threshold for Protein A binding)
    0.21,  # theta_B (expression threshold for Protein B binding)
    3,  # n_A (Hill coefficient for Protein A)
    3,  # n_B (Hill coefficient for Protein B)
    1.0,  # delta_PA (degradation rate of Protein A)
    1.0  # delta_PB (degradation rate of Protein B)
]

# Initial conditions
initial_conditions = [0.8, 0.8, 0.8, 0.8]  # [mRNA_A, mRNA_B, Protein_A, Protein_B]

# Time span for integration
t_span = (0, 100)  # Simulate for 10 seconds
t_eval = np.linspace(0, 100, 1000)  # Points to evaluate solution at

# Solve the ODE system
solution = solve_ivp(
    gene_regulation_ode,
    t_span,
    initial_conditions,
    args=(params,),
    method='RK45',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9
)

# Extract results
t = solution.t
mRNA_A = solution.y[0]
mRNA_B = solution.y[1]
Protein_A = solution.y[2]
Protein_B = solution.y[3]

# Plot (iii): Time evolution of mRNA A and mRNA B concentrations
plt.figure(figsize=(10, 6))
plt.plot(t, mRNA_A, 'b-', label='mRNA A')
plt.plot(t, mRNA_B, 'r-', label='mRNA B')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (M)')
plt.title('Time Evolution of mRNA A and mRNA B Concentrations')
plt.legend()
plt.grid(True)
plt.savefig('mrna_time_evolution.png')
plt.show()

# Plot (iv): Subspace plot of protein dynamics (phase-plane plot)
plt.figure(figsize=(8, 8))
plt.plot(Protein_A, Protein_B, 'g-')
plt.plot(Protein_A[0], Protein_B[0], 'go', label='Start')  # Start point
plt.plot(Protein_A[-1], Protein_B[-1], 'ro', label='End')  # End point
plt.xlabel('Protein A Concentration (M)')
plt.ylabel('Protein B Concentration (M)')
plt.title('Phase-Plane Plot of Protein A vs Protein B')
plt.grid(True)
plt.legend()
plt.savefig('protein_phase_plane.png')
plt.show()