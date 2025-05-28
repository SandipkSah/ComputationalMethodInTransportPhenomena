import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 0.050  # length of the rod
N = 6  # number of control volumes
dx = L / N
A = 1.0  # cross-sectional area
V = dx * A

T_inf = 500  # surrounding fluid temperature
T_init = 400.0  # initial guess

# Heat source parameters
a = 1e5  # W/m^3
b = 2e-3  # W/m^3/K^2

# Thermal conductivity parameters
k0 = 2.0  # W/mK
k1 = 0.002  # W/mK^2

# Heat transfer coefficients
hL = 50  # W/m2K
hR = 5   # W/m2K

# TDMA solver
def TDMA(aW, aP, aE, b):
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)
    T = np.zeros(n)

    # Forward sweep
    P[0] = aE[0] / aP[0]
    Q[0] = b[0] / aP[0]
    for i in range(1, n):
        denom = aP[i] - aW[i] * P[i - 1]
        P[i] = aE[i] / denom if i < n - 1 else 0
        Q[i] = (b[i] + aW[i] * Q[i - 1]) / denom

    # Back substitution
    T[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        T[i] = P[i] * T[i + 1] + Q[i]

    return T


def solve_problem_1():
    # Initialization
    # T = np.ones(N) * T_init
    T = np.array([470, 460, 450, 440, 430, 420 ])
    x = np.linspace(dx / 2, L - dx / 2, N)

    # Plotting setup
    plt.figure(figsize=(8, 6))

    # Iterate exactly 3 times
    for iteration in range(1, 4):
        kvals = k0 + k1 * (T - 400)
        S = a - b * (T - 400)**2
        SP = -2 * b * (T - 400)
        SC = S - SP * T

        aW = np.zeros(N)
        aE = np.zeros(N)
        aP = np.zeros(N)
        b_vec = np.zeros(N)

        # Coefficients assembly
        for i in range(N):
            if i > 0:
                kW = 0.5 * (kvals[i] + kvals[i - 1])
                aW[i] = kW * A / dx
            if i < N - 1:
                kE = 0.5 * (kvals[i] + kvals[i + 1])
                aE[i] = kE * A / dx

        for i in range(N):
            if i == 0:
                aP[i] = aE[i] + hL * A - SP[i] * V
                b_vec[i] = hL * A * T_inf + SC[i] * V
            elif i == N - 1:
                aP[i] = aW[i] + hR * A - SP[i] * V
                b_vec[i] = hR * A * T_inf + SC[i] * V
            else:
                aP[i] = aW[i] + aE[i] - SP[i] * V
                b_vec[i] = SC[i] * V
        # print(f"\nIteration {iteration} - b_vector:")
        # print(f"\nIteration {iteration} - aE: {aE}")
        # print(f"\nIteration {iteration} - aP: {aP}")
        # print(f"\nIteration {iteration} - aW: {aW}")
        

        # Solve system
        T = TDMA(aW, aP, aE, b_vec)

        # Print iteration result
        print(f"\nIteration {iteration} - Temperature Profile (K):")
        print(" ".join(format(val, ".3e") for val in T))

        print("SP\n=========", SP, "=========")
        print("SC\n=========", SC, "=========")
        print("b_vec\n=========", b_vec, "=========")

        # Plotting
        plt.plot(x, T, marker='o', label=f"Iteration {iteration}")

    # Final plot adjustments
    plt.xlabel("Rod Length (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Profile in 1D Porous Rod Over Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/temperature_profile_iterations.png")
    plt.show()


# Example usage
if __name__ == "__main__":
    solve_problem_1()
