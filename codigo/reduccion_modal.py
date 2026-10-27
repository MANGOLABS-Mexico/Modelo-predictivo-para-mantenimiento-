import numpy as np

def normalizar_estadistico(q_hat):
    """q_hat: (N, 4), muestras en las filas. Devuelve media y covarianza muestral (1/(N-1))."""
    mu = q_hat.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(q_hat, rowvar=False))
    return mu, Sigma

def _eigh_pd(Sigma, tol=1e-12):
    w2, U = np.linalg.eigh(Sigma)
    if w2.min() <= tol*w2.max():
        raise ValueError(f"Sigma no es definida positiva: eigenvalores entre {w2.min():.2e} y {w2.max():.2e}")
    return w2, U

def sqrt_sym(Sigma):
    w2, U = _eigh_pd(Sigma)
    return (U*np.sqrt(w2))@U.T

def inv_sqrt_sym(Sigma):
    w2, U = _eigh_pd(Sigma)
    return (U/np.sqrt(w2))@U.T

def proyectar_modal(mu, Sigma, Sq, Vr):
    """phi = Ts^-1 Vr con Ts = Sq Sigma^(1/2)  =>  phi = Sigma^(-1/2) Sq^(-1) Vr  (el orden importa).
    mu no interviene en phi; se conserva en la firma."""
    phi = inv_sqrt_sym(Sigma) @ np.diag(1.0/np.diag(Sq)) @ Vr
    Ppsi = np.linalg.inv(phi.T @ phi)
    return phi, Ppsi

def recuperar_q(Sq, mu, Vr, psi):
    """q = Sq mu + Vr psi, con psi la coordenada de la DESVIACIÓN (d = Sigma^-1/2 (q_hat - mu))."""
    return Sq@mu + Vr@psi