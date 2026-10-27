import numpy as np
from scipy.linalg import eigh

def construir_matrices(m, It, Ci, Ri, rCi, Cb, Rb, rb):
    n = 4
    sumC = sum(Ci); sumR = sum(Ri)
    sumCr = sum(Ci[i]*rCi[i] for i in range(n))
    sumRr = sum(Ri[i]*rCi[i] for i in range(n))
    sumCr2 = sum(Ci[i]*rCi[i]**2 for i in range(n))
    sumRr2 = sum(Ri[i]*rCi[i]**2 for i in range(n))
    M = np.diag([m, m, It, It])
    C = np.array([
        [Cb+sumC, 0, 0, Cb*rb+sumCr],
        [0, sumC, -sumCr, 0],
        [0, -sumCr, sumCr2, 0],
        [Cb*rb+sumCr, 0, 0, Cb*rb**2+sumCr2]
    ])
    K = np.array([
        [Rb+sumR, 0, 0, Rb*rb+sumRr],
        [0, sumR, -sumRr, 0],
        [0, -sumRr, sumRr2, 0],
        [Rb*rb+sumRr, 0, 0, Rb*rb**2+sumRr2]
    ])
    return M, C, K

def problema_eigenvalores(M, K, r=4):
    """w2, V de K v = w2 M v, con V^T M V = I. w tiene las unidades de M y K
    (si M y K son adimensionales, w es Omega = w_i/w0).
    Los eigenvectores salen con signo fijo: eigh no lo garantiza y el jacobiano numérico lo necesita."""
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(K)):
        raise ValueError("M o K no finitas")
    w2, V = eigh(K, M)                       # w2 ascendente
    if not np.all(np.isfinite(w2)):
        raise ValueError("eigh devolvió w2 no finito")
    if w2.min() <= 0.0:
        raise ValueError(f"K no es definida positiva: menor eigenvalor {w2.min():.3e}")
    for j in range(V.shape[1]):
        col = V[:, j]
        t = max(abs(col[0]), abs(col[1]))     # componente de traslación (y o z) dominante
        ref = (0 if abs(col[0]) >= abs(col[1]) else 1) if t > 1e-8*np.abs(col).max() else int(np.argmax(np.abs(col)))
        if col[ref] < 0:
            V[:, j] = -col
    return np.sqrt(w2[:r]), V[:, :r]

def amortiguamiento_rayleigh(aR, bR, M, K, V):
    """xi_i = c_i / (2 sqrt(k_i m_i)), con m_i=v'Mv, k_i=v'Kv, c_i=aR m_i + bR k_i.
    Vale con cualquier normalización de V; con V^T M V = I da xi_i = (aR/w_i + bR w_i)/2."""
    xi = np.zeros(V.shape[1])
    for i in range(V.shape[1]):
        vi = V[:, i]
        mi = vi @ M @ vi; ki = vi @ K @ vi
        xi[i] = (aR*mi + bR*ki)/(2*np.sqrt(ki*mi))
    return xi