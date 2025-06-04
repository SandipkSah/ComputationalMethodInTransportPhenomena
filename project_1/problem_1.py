import numpy as np
import matplotlib.pyplot as plt
import os

# --- Problem Data ---
L = 0.05
N = 5
dx = L / N
T_inf = 500
h_L = 50
h_R = 5
k0 = 2.0
k1 = 0.002
a = 1e5
b = 2.0

# --- Auxiliary Functions ---
def k(T):
    return k0 + k1 * (T - 400)

def S(T):
    if T < 400:
        return 1e5
    elif T > 600:
        return 0
    else:
        return a - b * (T - 400)**2

# --- TDMA Solver ---
def TDMA(A, B, C, D):
    n = len(D)
    P = np.zeros(n)
    Q = np.zeros(n)
    X = np.zeros(n)

    P[0] = C[0] / B[0]
    Q[0] = D[0] / B[0]
    for i in range(1, n):
        denom = B[i] - A[i] * P[i - 1]
        P[i] = C[i] / denom if i < n - 1 else 0
        Q[i] = (D[i] - A[i] * Q[i - 1]) / denom

    X[-1] = Q[-1]
    for i in reversed(range(n - 1)):
        X[i] = Q[i] - P[i] * X[i + 1]

    return X

def solve_problem_1():
    nodes = N + 1
    T = np.full(nodes, 400.0)
    max_iter = 100
    tol = 1e-3
    saved_iterations = [0, 1]  # Save iterations 1 and 2 (0-based)

    iteration_data = {}

    for it in range(max_iter):
        T_old = T.copy()
        A = np.zeros(nodes)
        B = np.zeros(nodes)
        C = np.zeros(nodes)
        D = np.zeros(nodes)

        for i in range(1, nodes - 1):
            k_e = 0.5 * (k(T[i]) + k(T[i + 1]))
            k_w = 0.5 * (k(T[i]) + k(T[i - 1]))
            A[i] = k_w / dx**2
            C[i] = k_e / dx**2
            B[i] = -A[i] - C[i]
            D[i] = -S(T[i])

        k_L_val = k(T[0])
        C[0] = k_L_val / dx
        B[0] = -(k_L_val / dx + h_L)
        D[0] = -h_L * T_inf

        k_R_val = k(T[-1])
        A[-1] = k_R_val / dx
        B[-1] = -(k_R_val / dx + h_R)
        D[-1] = -h_R * T_inf

        if it in saved_iterations:
            print(f"\n--- Iteration {it + 1} ---")
            for i in range(nodes):
                print(f"Node {i+1}: A = {A[i]:.3e}, B = {B[i]:.3e}, C = {C[i]:.3e}, D = {D[i]:.3e}")
            iteration_data[it + 1] = {
                'A': A.copy(),
                'B': B.copy(),
                'C': C.copy(),
                'D': D.copy(),
                'T': T.copy()
            }

        T = TDMA(A, B, C, D)
        if np.max(np.abs(T - T_old)) < tol:
            iteration_data['final'] = {
                'A': A.copy(),
                'B': B.copy(),
                'C': C.copy(),
                'D': D.copy(),
                'T': T.copy()
            }
            break

    print("\n--- Final Temperature Profile ---")
    for i, Ti in enumerate(T, 1):
        print(f"Node {i}: {Ti:.3f} K")

    # Plot
    x = np.linspace(0, L, nodes)
    plt.plot(x, T, marker='o')
    plt.title("Temperature Distribution in the Rod")
    plt.xlabel("Position along the rod [m]")
    plt.ylabel("Temperature [K]")
    plt.grid(True)
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/problem_1_temperature_profile_iterations.png")
    plt.show()

    return iteration_data

# Run and get iteration data
if __name__ == "__main__":
    data = solve_problem_1()
