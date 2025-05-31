import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
L = 0.05       # Rod length (m)
A = 1.0        # Cross-sectional area (m²)
T0 = 400.0     # Initial temperature (K)
T_inf = 500    # Surrounding fluid temperature (K)
rho = 1500     # Density (kg/m³)
Cp = 1000      # Specific heat (J/kg·K)
a = 1e5        # W/m³
b = 2e-3       # W/m³/K²
k0 = 2.0       # W/mK
k1 = 0.002     # W/mK²
hL = 50        # Left boundary heat transfer coefficient
hR = 5         # Right boundary heat transfer coefficient

N = 12
dx = L / N
V = A * dx
dt = 0.1
max_time = 2000.0
steady_tol = 1e-3
max_steps = int(max_time / dt)

# TDMA solver
def TDMA(aW, aP, aE, b):
    n = len(b)
    P, Q, T = np.zeros(n), np.zeros(n), np.zeros(n)
    P[0] = aE[0] / aP[0]
    Q[0] = b[0] / aP[0]
    for i in range(1, n):
        denom = aP[i] - aW[i] * P[i - 1]
        P[i] = aE[i] / denom if i < n - 1 else 0
        Q[i] = (b[i] + aW[i] * Q[i - 1]) / denom
    T[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        T[i] = P[i] * T[i + 1] + Q[i]
    return T

# Heat source
def compute_source(T):
    S = np.zeros_like(T)
    S[T < 400] = a
    mid_mask = (T >= 400) & (T <= 600)
    S[mid_mask] = a - b * (T[mid_mask] - 400)**2
    return S

# Transient simulation
def transient_simulation():
    T = np.ones(N) * T0
    time = 0.0
    times, history, center_history, delta_Ts = [0.0], [T.copy()], [T[N // 2]], []
    steady_time = None

    for step in range(max_steps):
        kvals = k0 + k1 * (T - 400)
        S = compute_source(T)

        aW = np.zeros(N)
        aE = np.zeros(N)
        aP = np.zeros(N)
        b_vec = np.zeros(N)

        for i in range(N):
            if i > 0:
                kW = 0.5 * (kvals[i] + kvals[i - 1])
                aW[i] = kW * A / dx
            if i < N - 1:
                kE = 0.5 * (kvals[i] + kvals[i + 1])
                aE[i] = kE * A / dx

        for i in range(N):
            if i == 0:
                aP[i] = aE[i] + hL * A + rho * Cp * V / dt
                b_vec[i] = hL * A * T_inf + rho * Cp * V * T[i] / dt + S[i] * V
            elif i == N - 1:
                aP[i] = aW[i] + hR * A + rho * Cp * V / dt
                b_vec[i] = hR * A * T_inf + rho * Cp * V * T[i] / dt + S[i] * V
            else:
                aP[i] = aW[i] + aE[i] + rho * Cp * V / dt
                b_vec[i] = rho * Cp * V * T[i] / dt + S[i] * V

        T_new = TDMA(aW, aP, aE, b_vec)
        delta_T = np.max(np.abs(T_new - T))

        delta_Ts.append(delta_T)
        center_history.append(T_new[N // 2])
        time += dt
        times.append(time)
        history.append(T_new.copy())

        if step % 100 == 0:
            print(f"Step {step:5d} | Time: {time:6.1f} s | Center T: {T_new[N//2]:.2f} K | Max ΔT: {delta_T:.5f}")

        if delta_T < steady_tol and steady_time is None:
            steady_time = time
            print(f"\n✅ Reached steady-state at t = {steady_time:.2f} s (ΔT < {steady_tol})")
            break

        T = T_new

    if steady_time is None:
        print("\n⚠️ Steady-state not reached within max simulation time.")

    return np.array(times), np.array(history), T, delta_Ts, steady_time

# Problem 4 wrapper
def solve_problem_4():
    os.makedirs("output", exist_ok=True)

    times, temps, final_T, delta_Ts, steady_time = transient_simulation()

    print("\nFinal temperature profile (K):")
    print(" ".join("{:.3f}".format(t) for t in final_T))

    # Plot center temperature
    center_Ts = [T[N // 2] for T in temps]
    plt.figure()
    plt.plot(times, center_Ts, label="Center Temperature")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Problem 4: Center Temperature vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/problem_4_center_temperature_vs_time.png")

    # Plot convergence
    plt.figure()
    plt.semilogy(times[1:], delta_Ts, label="Max ΔT")
    plt.xlabel("Time (s)")
    plt.ylabel("Max ΔT (K)")
    plt.title("Problem 4: Convergence to Steady-State")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/problem_4_convergence.png")
    plt.show()

    print("✅ Plots saved to output/")

# Optional direct execution
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    solve_problem_4()
