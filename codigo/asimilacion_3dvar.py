import numpy as np

def ganancia(Bb, H, Ro):
    S = H @ Bb @ H.T + Ro
    return np.linalg.solve(S, H @ Bb).T          # K = Bb H^T S^{-1}  (S y Bb simétricas)

def analisis_lineal(sb, Bb, H, c, yo, Ro):
    K = ganancia(Bb, H, Ro)
    sa = sb + K @ (yo - c - H @ sb)
    Pa = (np.eye(Bb.shape[0]) - K @ H) @ Bb
    Pa = 0.5 * (Pa + Pa.T)
    return sa, Pa

def construir_H(Omega, xi, Vr, T, w0):
    """Omega = w_i/w0 (adimensional); xi = factor de amortiguamiento de cada modo."""
    Omega2 = np.diag(Omega**2)
    D = np.diag(2*xi*Omega)
    G = T @ Vr
    return -w0**2 * G @ np.hstack([Omega2, D])

def construir_Bb(Ppsi, Omegaf, sigma_b, r):
    B = np.zeros((2*r, 2*r))
    B[:r, :r] = sigma_b**2 * Ppsi
    B[r:, r:] = sigma_b**2 * Omegaf**2 * Ppsi
    return B