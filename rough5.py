import numpy as np
import matplotlib.pyplot as plt
import os

# === Physical Parameters ===
L = 0.05        # Length of the rod [m]
T_inf = 500     # External fluid temperature [K]
h_L = 50        # Left convection coefficient [W/m²K]
h_R = 5         # Right convection coefficient [W/m²K]
k0 = 2.0        # Base thermal conductivity [W/mK]
k1 = 0.002      # Thermal coefficient [W/mK²]
a = 1e5         # Heat source constant [W/m³]
b = 2.0         # Source coefficient [W/m³K²]

# === Physical Functions ===
def k(T):
    return k0 + k1 * (T - 400)

def S(T):
    if T < 400:
        return 1e6
    elif T > 600:
        return 0
    else:
        return a - b * (T - 400)**2

# === TDMA Algorithm ===
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

# === Steady-state Solver ===
def solve_temperature_profile(N, max_iter=100, tol=1e-3):
    dx = L / N
    nodes = N + 1
    T = np.full(nodes, 400.0)
    for _ in range(max_iter):
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
        # Boundary conditions with convection
        kL = k(T[0])
        C[0] = kL / dx
        B[0] = -(C[0] + h_L)
        D[0] = -h_L * T_inf
        kR = k(T[-1])
        A[-1] = kR / dx
        B[-1] = -(A[-1] + h_R)
        D[-1] = -h_R * T_inf
        T = TDMA(A, B, C, D)
        if np.max(np.abs(T - T_old)) < tol:
            break
    return T

def solve_problem_2():
        
    # === Solve for 6, 12, 24 Control Volumes ===
    T6 = solve_temperature_profile(6)
    T12 = solve_temperature_profile(12)
    T24 = solve_temperature_profile(24)

    # Extract center temperatures
    center6 = len(T6) // 2
    center12 = len(T12) // 2
    center24 = len(T24) // 2

    T6_center = T6[center6]
    T12_center = T12[center12]
    T24_center = T24[center24]

    # === Richardson Extrapolation (p = 2) ===
    T_exact = T12_center + (T12_center - T6_center) / (2**2 - 1)

    # === Relative Error of 6 CVs ===
    rel_error = abs(T6_center - T_exact) / abs(T_exact) * 100

    # === Display Results ===
    print("Center temperature for different meshes:")
    print(f"6 CVs   = {T6_center:.3f} K")
    print(f"12 CVs  = {T12_center:.3f} K")
    print(f"24 CVs  = {T24_center:.3f} K")
    print(f"Richardson Extrapolated ≈ {T_exact:.3f} K")
    print(f"\nRelative Error (6 CVs vs. Exact) = {rel_error:.4f} %")

    # === Plot the Profiles ===
    x6 = np.linspace(0, L, len(T6)) * 100      # cm
    x12 = np.linspace(0, L, len(T12)) * 100
    x24 = np.linspace(0, L, len(T24)) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(x6, T6, 'o-', label='6 CVs')
    plt.plot(x12, T12, 's-', label='12 CVs')
    plt.plot(x24, T24, '^-', label='24 CVs')
    plt.xlabel('Position along the rod (cm)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Profile Comparison - 6, 12, and 24 CVs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Report Summary Output ===
    print("\n=== REPORT SUMMARY ===")
    print(f"Rod Length: {L:.3f} m")
    print(f"External Fluid Temperature: {T_inf:.1f} K")
    print(f"Convection Coefficients: Left = {h_L} W/m²K, Right = {h_R} W/m²K")
    print(f"Thermal Conductivity: k(T) = {k0} + {k1}(T - 400) W/mK")
    print(f"Heat Source: S(T) = {a} - {b}(T - 400)^2 W/m³ (for 400K ≤ T ≤ 600K)")
    print(f"Number of Control Volumes: 6, 12, 24\n")

    print("Center Temperatures:")
    print(f"  • 6 CVs   = {T6_center:.3f} K")
    print(f"  • 12 CVs  = {T12_center:.3f} K")
    print(f"  • 24 CVs  = {T24_center:.3f} K")
    print(f"  • Richardson Extrapolated ≈ {T_exact:.3f} K")
    print(f"  • Relative Error (6 CVs vs. Extrapolated) = {rel_error:.4f} %\n")

    print("Temperature Profile Arrays:")
    print(f"T6  = {np.round(T6, 3)}")
    print(f"T12 = {np.round(T12, 3)}")
    print(f"T24 = {np.round(T24, 3)}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    solve_problem_2()
