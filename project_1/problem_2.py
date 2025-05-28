import numpy as np

# Physical constants
L = 0.05  # rod length in meters
A = 1.0  # cross-sectional area (assumed 1 m²)
a = 1e5  # W/m³
b = 2e-3  # W/m³/K²
k0 = 2.0  # W/mK
k1 = 0.002  # W/mK²
T_inf = 500  # surrounding fluid temperature in K
hL = 50  # W/m²K
hR = 5   # W/m²K
T_init = 400.0
max_iter = 100

# TDMA solver
def TDMA(aW, aP, aE, b):
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)
    T = np.zeros(n)

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

# Piecewise source function
def compute_source(T):
    S = np.zeros_like(T)
    mask_low = T < 400
    mask_mid = (T >= 400) & (T <= 600)
    mask_high = T > 600

    S[mask_low] = a
    S[mask_mid] = a - b * (T[mask_mid] - 400)**2
    S[mask_high] = 0
    return S

# Linearized source
def compute_SP(T):
    SP = np.zeros_like(T)
    mask_mid = (T >= 400) & (T <= 600)
    SP[mask_mid] = -2 * b * (T[mask_mid] - 400)
    return SP

# Solve for N CVs
def solve_temperature_profile(N):
    dx = L / N
    V = dx * A
    T = np.ones(N) * T_init

    for _ in range(max_iter):
        kvals = k0 + k1 * (T - 400)
        S = compute_source(T)
        SP = compute_SP(T)
        SC = S - SP * T

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
                aP[i] = aE[i] + hL * A - SP[i] * V
                b_vec[i] = hL * A * T_inf + SC[i] * V
            elif i == N - 1:
                aP[i] = aW[i] + hR * A - SP[i] * V
                b_vec[i] = hR * A * T_inf + SC[i] * V
            else:
                aP[i] = aW[i] + aE[i] - SP[i] * V
                b_vec[i] = SC[i] * V

        T = TDMA(aW, aP, aE, b_vec)

    return T

# Richardson extrapolation
def richardson_extrapolation(T12, T24, dx12, dx24):
    i12 = len(T12) // 2
    i24 = len(T24) // 2
    return T24[i24] + (T24[i24] - T12[i12]) / ((dx12 / dx24)**2 - 1)

# Problem 2 runner
def solve_problem_2():
    T12 = solve_temperature_profile(12)
    T24 = solve_temperature_profile(24)

    dx12 = L / 12
    dx24 = L / 24
    Texact = richardson_extrapolation(T12, T24, dx12, dx24)

    print("Temperature profile (12 CVs):")
    print(" ".join("{:.3e}".format(val) for val in T12))

    print("\nTemperature profile (24 CVs):")
    print(" ".join("{:.3e}".format(val) for val in T24))

    print("\nEstimated center temperature (Richardson extrapolation): {:.3e} K".format(Texact))


# Example usage
if __name__ == "__main__":
    solve_problem_2()
