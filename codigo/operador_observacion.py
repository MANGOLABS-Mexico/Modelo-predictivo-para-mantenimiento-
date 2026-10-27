import numpy as np

def matriz_T(rCi, n_o=8):
    """n_o=8: canales (y_C1, z_C1, ..., y_C4, z_C4). n_o=4: un canal por cojinete, supuesto en y
    (la descripción del dataset no dice la dirección; con las 4 en y, rank(T)=2)."""
    if n_o == 8:
        T = np.zeros((8, 4))
        for i in range(4):
            T[2*i, :] = [1, 0, 0, rCi[i]]
            T[2*i+1, :] = [0, 1, -rCi[i], 0]
    elif n_o == 4:
        T = np.zeros((4, 4))
        for i in range(4):
            T[i, :] = [1, 0, 0, rCi[i]]
    else:
        raise ValueError("n_o debe ser 8 o 4")
    return T

def aceleraciones_a_qddot(a, T):
    """a: (N, n_o), muestras en las filas y canales en las columnas  ->  qddot: (N, 4)."""
    if a.ndim != 2 or a.shape[1] != T.shape[0]:
        raise ValueError(f"a debe ser (N, {T.shape[0]}); recibido {a.shape}")
    return a @ np.linalg.pinv(T).T

def integrar_a_desplazamiento(qddot, fs, fc=20.0):
    """qddot: (N, 4). Doble integración en frecuencia: cada componente de Fourier de frecuencia f
    se divide por -(2 pi f)^2 si f >= fc y se anula si f < fc (fc en Hz, por debajo de la frecuencia de giro)."""
    if qddot.ndim != 2 or qddot.shape[0] < qddot.shape[1]:
        raise ValueError(f"qddot debe ser (N, 4) con muestras en las filas; recibido {qddot.shape}")
    n = qddot.shape[0]
    f = np.fft.rfftfreq(n, 1/fs)
    factor = np.zeros_like(f)
    mask = f >= fc
    factor[mask] = -1.0/((2*np.pi*f[mask])**2)
    X = np.fft.rfft(qddot, axis=0)
    return np.fft.irfft(X*factor[:, None], n=n, axis=0)