import numpy as np
import matplotlib.pyplot as plt
import os

# --- Constants ---
L = 0.05       # Rod length (m)
A = 1.0        # Cross-sectional area (m²)
T0 = 400.0     # Initial temperature (K)
T_inf = 500    # Surrounding fluid temperature (K)
rho = 1500     # Density (kg/m³)
Cp = 1000      # Specific heat (J/kg·K)
a = 1e5        # Heat source constant (W/m³)
b = 2e-3       # Heat source temp coefficient (W/m³/K²)
k0 = 2.0       # Base thermal conductivity (W/mK)
k1 = 0.002     # Thermal conductivity temp coefficient (W/mK²)
hL = 50        # Left convective heat transfer coefficient (W/m²K)
hR = 5         # Right convective heat transfer coefficient (W/m²K)

# --- TDMA solver ---
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

# --- Heat source function ---
def compute_source(T):
    S = np.zeros_like(T)
    S[T < 400] = a
    mask_mid = (T >= 400) & (T <= 600)
    S[mask_mid] = a - b * (T[mask_mid] - 400)**2
    return S

# --- Transient simulation using implicit Euler method ---
def transient_simulation(N=12, dt=0.1, max_steps=100000, center_tol=0.1, window=100):
    dx = L / N
    V = A * dx
    T = np.ones(N) * T0
    time = 0.0
    history, times, center_history = [T.copy()], [0.0], [T[N // 2]]

    for step in range(max_steps):
        kvals = k0 + k1 * (T - 400)
        S = compute_source(T)

        aW, aE, aP, b_vec = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            if i > 0:
                kW = 0.5 * (kvals[i] + kvals[i - 1])
                aW[i] = kW * A / dx
            if i < N - 1:
                kE = 0.5 * (kvals[i] + kvals[i + 1])
                aE[i] = kE * A / dx

        for i in range(N):
            if i == 0:
                k_face = 0.5 * (kvals[i] + kvals[i + 1])
                aE[i] = k_face * A / dx
                aP[i] = aE[i] + hL * A + rho * Cp * V / dt
                b_vec[i] = hL * A * T_inf + rho * Cp * V * T[i] / dt + S[i] * V
            elif i == N - 1:
                k_face = 0.5 * (kvals[i] + kvals[i - 1])
                aW[i] = k_face * A / dx
                aP[i] = aW[i] + hR * A + rho * Cp * V / dt
                b_vec[i] = hR * A * T_inf + rho * Cp * V * T[i] / dt + S[i] * V
            else:
                aP[i] = aW[i] + aE[i] + rho * Cp * V / dt
                b_vec[i] = rho * Cp * V * T[i] / dt + S[i] * V

        T_new = TDMA(aW, aP, aE, b_vec)
        delta_T = np.max(np.abs(T_new - T))
        center_T = T_new[N // 2]

        if step % 100 == 0:
            print(f"Step {step:5d} | Time: {time:6.1f} s | Center T: {center_T:.3e} K | Max ΔT: {delta_T:.4e}")

        center_history.append(center_T)
        if len(center_history) > window:
            delta_center = abs(center_history[-1] - center_history[-window])
            if delta_center < center_tol:
                print(f"\n✅ Converged: Center temperature stabilized within {center_tol} K over {window*dt:.1f} s.")
                break

        T = T_new
        time += dt
        history.append(T.copy())
        times.append(time)

    return np.array(times), np.array(history), T

# --- Solve and plot ---
def solve_problem_3():
    os.makedirs("output", exist_ok=True)
    times, temps, final_T = transient_simulation()

    print("\nFinal temperature profile (K):")
    print(" ".join("{:.3e}".format(val) for val in final_T))

    center_Ts = [t[len(t)//2] for t in temps]
    plt.figure()
    plt.plot(times, center_Ts, label="Center Temperature")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Problem 3: Center Temperature vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/problem_3_center_temperature_vs_time.png")
    plt.show()
    print("✅ Plot saved to output/problem_3_center_temperature_vs_time.png")

# --- Run the simulation ---
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    solve_problem_3()
