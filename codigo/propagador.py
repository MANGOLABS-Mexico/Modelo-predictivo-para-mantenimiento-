import numpy as np
from scipy.linalg import expm

def bloque_propagador(w, xi, dt, tol_critico=1e-12):
    """Propagador exacto de x'' + 2 xi w x' + w^2 x = 0 sobre el estado [x, x'] en un paso dt."""
    if w <= 0 or xi <= 0:
        raise ValueError(f"se requiere w>0 y xi>0; recibido w={w}, xi={xi}")
    a = xi*w
    if xi < 1.0 - tol_critico:                       # subamortiguado
        wd = w*np.sqrt(1.0 - xi**2)
        e = np.exp(-a*dt); c = np.cos(wd*dt); s = np.sin(wd*dt)
        return np.array([[e*(c + a/wd*s), e*s/wd],
                         [-e*(w**2/wd)*s, e*(c - a/wd*s)]])
    if xi > 1.0 + tol_critico:                       # sobreamortiguado
        k = w*np.sqrt(xi**2 - 1.0)
        x = k*dt
        if x < 30.0:
            e = np.exp(-a*dt); ch = e*np.cosh(x); sh = e*np.sinh(x)
        else:                                        # evita desbordamiento de cosh/sinh
            e1 = np.exp((-a + k)*dt); e2 = np.exp((-a - k)*dt)
            ch = 0.5*(e1 + e2); sh = 0.5*(e1 - e2)
        return np.array([[ch + a/k*sh, sh/k],
                         [-(w**2/k)*sh, ch - a/k*sh]])
    e = np.exp(-a*dt)                                # crítico
    return np.array([[e*(1 + a*dt), e*dt],
                     [-e*w**2*dt, e*(1 - a*dt)]])

def propagador_total(w, xi, dt):
    """Phi sobre el estado s=[psi_1..psi_r, psi'_1..psi'_r] (el orden de H y de B_b)."""
    P = [bloque_propagador(w[i], xi[i], dt) for i in range(len(w))]
    return np.block([[np.diag([p[0, 0] for p in P]), np.diag([p[0, 1] for p in P])],
                     [np.diag([p[1, 0] for p in P]), np.diag([p[1, 1] for p in P])]])

def propagador_general(A, dt):
    return expm(A*dt)