import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
L = 0.05  # rod length in meters
N = 12  # number of control volumes
dx = L / N
rho = 1500  # kg/m^3
Cp = 1000  # J/kg.K
T_init = 400.0  # K
T_fluid_L = 500.0  # K
T_fluid_R = 500.0  # K
h_L = 50.0  # W/m2K
h_R = 5.0  # W/m2K
a_source = 100000.0  # W/m3
b_source = 2.0  # W/m3.K^2
k0 = 2.0  # W/m.K
k1 = 0.002  # W/m.K^2
dt = 0.25  # time step in seconds
tol = 1e-2  # convergence tolerance
max_time = 5000  # max time in seconds

# --- Functions ---
def compute_k(T):
    return k0 + k1 * (T - 400.0)

def compute_S(T):
    if T < 400:
        return a_source
    elif T > 600:
        return 0.0
    else:
        return a_source - b_source * (T - 400.0) ** 2

def tdma_solver(a_w, a_p, a_e, b):
    N = len(b)
    P = np.zeros(N)
    Q = np.zeros(N)
    T = np.zeros(N)

    P[0] = a_e[0] / a_p[0]
    Q[0] = b[0] / a_p[0]

    for i in range(1, N):
        denom = a_p[i] - a_w[i] * P[i - 1]
        P[i] = a_e[i] / denom
        Q[i] = (b[i] + a_w[i] * Q[i - 1]) / denom

    T[-1] = Q[-1]
    for i in range(N - 2, -1, -1):
        T[i] = P[i] * T[i + 1] + Q[i]
    return T

# --- Initialization ---
T = np.full(N, T_init)
time = 0.0

# --- Time Loop ---
while time < max_time:
    T_old = T.copy()
    a_w = np.zeros(N)
    a_e = np.zeros(N)
    a_p = np.zeros(N)
    b_vec = np.zeros(N)

    for i in range(N):
        T_P = T_old[i]
        S = compute_S(T_P)
        k_P = compute_k(T_P)

        Sc = S
        Sp = 0.0

        if i == 0:
            k_e = compute_k((T_old[i] + T_old[i + 1]) / 2)
            a_e[i] = k_e / dx
            a_w[i] = 0.0
            a_p[i] = a_e[i] + h_L + rho * Cp * dx / dt - Sp * dx
            b_vec[i] = rho * Cp * dx * T_old[i] / dt + Sc * dx + h_L * T_fluid_L
        elif i == N - 1:
            k_w = compute_k((T_old[i] + T_old[i - 1]) / 2)
            a_w[i] = k_w / dx
            a_e[i] = 0.0
            a_p[i] = a_w[i] + h_R + rho * Cp * dx / dt - Sp * dx
            b_vec[i] = rho * Cp * dx * T_old[i] / dt + Sc * dx + h_R * T_fluid_R
        else:
            k_w = compute_k((T_old[i] + T_old[i - 1]) / 2)
            k_e = compute_k((T_old[i] + T_old[i + 1]) / 2)
            a_w[i] = k_w / dx
            a_e[i] = k_e / dx
            a_p[i] = a_w[i] + a_e[i] + rho * Cp * dx / dt - Sp * dx
            b_vec[i] = rho * Cp * dx * T_old[i] / dt + Sc * dx

    T = tdma_solver(a_w, a_p, a_e, b_vec)
    time += dt

    if np.max(np.abs(T - T_old)) < tol:
        break

# --- Plot Result ---
x = np.linspace(0, L, N)
plt.plot(x, T, marker='o')
plt.xlabel('Position along rod [m]')
plt.ylabel('Temperature [K]')
plt.title(f'Steady-state Profile (t ≈ {time:.1f} s)')
plt.grid(True)
plt.show()

# Print the steady-state time
print(f"Time to reach steady state: {time:.1f} seconds")
