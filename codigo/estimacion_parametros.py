import warnings
import numpy as np

def h_observacion(z, r, parametros_fn):
    s = z[:2*r]
    theta = z[2*r:2*r+7]
    eta = z[2*r+7:]
    H, c = parametros_fn(theta, eta)
    return H @ s + c

def jacobiano_numerico(z, r, parametros_fn, eps=1e-4):
    n = z.size
    h0 = h_observacion(z, r, parametros_fn)
    J = np.zeros((h0.size, n))
    for j in range(n):
        zp = z.copy()
        paso = eps*max(abs(zp[j]), 1.0)
        zp[j] += paso
        J[:, j] = (h_observacion(zp, r, parametros_fn) - h0)/paso
    return J

def gauss_newton(zb, Bz, yo, Ro, r, parametros_fn, max_iter=20, tol=1e-6):
    """Minimiza J(z)=1/2(z-zb)'Bz^-1(z-zb)+1/2(yo-h(z))'Ro^-1(yo-h(z)).
    Iteración del documento: z^(j+1) = zb + Bz J' (J Bz J' + Ro)^-1 [yo - h(z^j) + J (z^j - zb)],
    con retroceso: el paso se acepta solo si J baja; si no, se divide a la mitad."""
    Bi = np.linalg.inv(Bz); Ri = np.linalg.inv(Ro)
    def costo(z):
        try:
            h = h_observacion(z, r, parametros_fn)
        except Exception:
            return np.inf
        if not np.all(np.isfinite(h)):
            return np.inf
        d = z - zb; e = yo - h
        return 0.5*d@Bi@d + 0.5*e@Ri@e
    z = zb.copy(); Jz = costo(z)
    if not np.isfinite(Jz):
        warnings.warn("gauss_newton: h(zb) no es finito; se devuelve zb")
        return zb
    for _ in range(max_iter):
        h = h_observacion(z, r, parametros_fn)
        J = jacobiano_numerico(z, r, parametros_fn)
        S = J@Bz@J.T + Ro
        z_gn = zb + Bz@J.T@np.linalg.solve(S, yo - h + J@(z - zb))
        paso = z_gn - z
        alpha = 1.0
        while alpha > 1e-4:
            zn = z + alpha*paso; Jn = costo(zn)
            if Jn < Jz: break
            alpha *= 0.5
        else:
            break                      # sin descenso: z es (casi) un mínimo local
        z, Jz = zn, Jn
        if np.linalg.norm(alpha*paso) < tol*(1 + np.linalg.norm(z)):
            break
    return z