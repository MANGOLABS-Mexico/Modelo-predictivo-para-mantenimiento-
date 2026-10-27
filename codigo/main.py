import numpy as np
import os
from cargar_datos import cargar_todo_conjunto1, FS
from modelo_fisico import construir_matrices, problema_eigenvalores, amortiguamiento_rayleigh
from operador_observacion import matriz_T, aceleraciones_a_qddot, integrar_a_desplazamiento
from reduccion_modal import normalizar_estadistico, proyectar_modal
from propagador import propagador_total
from asimilacion_3dvar import construir_H, construir_Bb
from estimacion_parametros import gauss_newton
from graficos_prediccion import (graficar_R_normalizado, graficar_xi, graficar_w, graficar_C_rayleigh,
                                 heatmap_espectro_canales, graficar_alarma)
import matplotlib.pyplot as plt
DIR_SALIDA = os.path.dirname(os.path.abspath(__file__))

m = 10.0; r_cil = 0.02; l_cil = 0.3
It = m*(r_cil**2/4 + l_cil**2/12)
L = 0.05
rCi0 = np.array([-0.15, -0.05, 0.05, 0.15]); rb = -0.20
Ri0 = np.array([1e6]*4); Ci0 = np.array([1e3]*4); Rb0 = 5e5; Cb0 = 5e2
aR0, bR0 = 0.5, 1e-5                       # Rayleigh nominal
w0 = 2*np.pi*2000/60                       # frecuencia del motor
r = 4
SIGMA_B, SIGMA_THETA, SIGMA_F = 1.0, 0.5, 3e-3
NC = 600                                   # muestras consecutivas por archivo (una vuelta del eje)
paso = 20

M, C, K = construir_matrices(m, It, Ci0, Ri0, rCi0, Cb0, Rb0, rb)
w, V = problema_eigenvalores(M, K, r=r)

datos, tiempos = cargar_todo_conjunto1()
T0 = matriz_T(rCi0, n_o=8)
a_ref = datos[0].T                                        # (N, 8)
q_ref = integrar_a_desplazamiento(aceleraciones_a_qddot(a_ref, T0), FS, fc=20.0)
Sq = np.diag([L, L, 1.0, 1.0])
q_hat = q_ref @ np.linalg.inv(Sq).T
mu, Sigma = normalizar_estadistico(q_hat)
phi, Ppsi = proyectar_modal(mu, Sigma, Sq, V[:, :r])

Ro = np.eye(8)*1e-4
Bb = construir_Bb(Ppsi, (2*np.pi*2000/60)/w0, SIGMA_B, r)              # Omega_f = 1
B_theta = np.eye(7)*SIGMA_THETA**2
B_eta = np.diag([0.3**2] + [0.01**2]*5 + [SIGMA_F**2]*2)
Bz = np.block([[Bb, np.zeros((2*r, 15))],
               [np.zeros((7, 2*r)), B_theta, np.zeros((7, 8))],
               [np.zeros((8, 2*r)), np.zeros((8, 7)), B_eta]])

def modelo(theta, eta):
    Ri = Ri0*np.exp(theta[:4]); Rb = Rb0*np.exp(theta[4])
    aR = aR0*np.exp(theta[5]); bR = bR0*np.exp(theta[6])            # centrados en los nominales
    rC = rCi0 + L*eta[1:5]; rbb = rb + L*eta[5]                      # eta = desviación respecto al nominal, en unidades de L
    M_, C_, K_ = construir_matrices(m, It*np.exp(eta[0]), Ci0, Ri, rC, Cb0, Rb, rbb)
    w_, V_ = problema_eigenvalores(M_, K_, r=r)                      # lanza error si K no es definida positiva
    xi_ = amortiguamiento_rayleigh(aR, bR, M_, K_, V_)
    F = eta[6:8]*(m*L*w0**2)                                         # F = F~ m L w0^2
    b1 = np.array([1, 0, 0, rC[1]]); b2 = np.array([1, 0, 0, rC[2]])
    return dict(M=M_, w=w_, V=V_, xi=xi_, T=matriz_T(rC, 8), f=-(b1*F[0] + b2*F[1]), Ri=Ri)

def parametros_fn(theta, eta):
    d = modelo(theta, eta)
    H = construir_H(d['w']/w0, d['xi'], d['V'], d['T'], w0)          # Omega = w/w0
    c = d['T'] @ np.linalg.solve(d['M'], d['f'])
    return H, c

def pronostico(s_a, theta, eta):
    d = modelo(theta, eta); Om = d['w']/w0
    Phi = propagador_total(Om, d['xi'], w0/FS)                       # Delta tau = w0 * dt
    u = d['V'].T @ d['f']/w0**2
    s_e = np.concatenate([u/Om**2, np.zeros(r)])
    return Phi @ s_a + (np.eye(2*r) - Phi) @ s_e, s_e

theta_a = np.zeros(7); eta_a = np.zeros(8)
R_hist, Rb_hist, w_hist, xi_hist, t_hist = [], [], [], [], []
aR_hist, bR_hist = [], []


for k in range(0, len(datos), paso):
    ymed = datos[k].T
    _, s_a = pronostico(np.zeros(2*r), theta_a, eta_a)               # inicio de archivo: equilibrio forzado
    for n in range(NC):
        s_b, _ = pronostico(s_a, theta_a, eta_a)
        zb = np.concatenate([s_b, theta_a, eta_a])
        z_a = gauss_newton(zb, Bz, ymed[n, :], Ro, r, parametros_fn, max_iter=3)
        s_a = z_a[:2*r]; theta_a = z_a[2*r:2*r+7]; eta_a = z_a[2*r+7:]
    d = modelo(theta_a, eta_a)
    R_hist.append(d['Ri'].copy()); Rb_hist.append(Rb0*np.exp(theta_a[4]))
    w_hist.append(d['w'].copy()); xi_hist.append(d['xi'].copy()); t_hist.append(tiempos[k])
   
    aR_hist.append(aR0*np.exp(theta_a[5])); bR_hist.append(bR0*np.exp(theta_a[6]))


    print("archivo", k, "de", len(datos), "R/R0 =", np.round(d['Ri']/Ri0, 3), "xi =", np.round(d['xi'], 4))

R_arr = np.array(R_hist); xi_arr = np.array(xi_hist)
for nombre, arr in [("R_hist", R_arr), ("Rb_hist", np.array(Rb_hist)), ("w_hist", np.array(w_hist)),
                    ("xi_hist", xi_arr), ("t_hist", np.array(t_hist)),("aR_hist", np.array(aR_hist)), 
                    ("bR_hist", np.array(bR_hist))]:
    np.save(os.path.join(DIR_SALIDA, nombre + ".npy"), arr)





graficar_R_normalizado(t_hist, R_arr, ruta=os.path.join(DIR_SALIDA, "R_normalizado.pdf"))
graficar_w(t_hist, np.array(w_hist), ruta=os.path.join(DIR_SALIDA, "w.pdf"))
graficar_C_rayleigh(t_hist, aR_hist, bR_hist, np.array(w_hist), ruta=os.path.join(DIR_SALIDA, "C_rayleigh.pdf"))
graficar_xi(t_hist, xi_arr, ruta=os.path.join(DIR_SALIDA, "xi.pdf"))
heatmap_espectro_canales(datos[::paso], t_hist, [4, 5], ruta=os.path.join(DIR_SALIDA, "heatmap_coj3.pdf"),
                         titulo="Cojinete 3 (canales 5,6) - BPFI=297Hz")
heatmap_espectro_canales(datos[::paso], t_hist, [6, 7], ruta=os.path.join(DIR_SALIDA, "heatmap_coj4.pdf"),
                         titulo="Cojinete 4 (canales 7,8) - BSF=139Hz, 2xBSF=278Hz")
graficar_alarma(t_hist, R_arr, ruta=os.path.join(DIR_SALIDA, "alarma.pdf"))
print("R/R0 final =", R_arr[-1]/Ri0, "  xi final =", xi_arr[-1])
plt.show()