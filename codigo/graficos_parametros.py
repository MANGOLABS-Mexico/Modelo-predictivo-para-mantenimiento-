import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _fechas(tiempos):
    return np.array(tiempos,dtype="datetime64[s]")

def graficar_evolucion_rigideces(tiempos,R_hist,Rb_hist,ruta=None):
    t=_fechas(tiempos)
    R_hist=np.array(R_hist)
    Rb_hist=np.array(Rb_hist).ravel()
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for i in range(R_hist.shape[1]):
        axs[0].plot(t,R_hist[:,i],label="R_%d"%(i+1),marker=".",ms=3)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("R_i [N/m]")
    axs[0].legend(ncol=4)
    axs[0].set_title("Rigidez equivalente de cojinetes")
    axs[1].plot(t,Rb_hist,marker=".",ms=3,color="k")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("R_b [N/m]")
    axs[1].set_xlabel("tiempo")
    axs[1].set_title("Rigidez equivalente de banda")
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    axs[1].xaxis_date()
    plt.setp(axs[1].get_xticklabels(),rotation=30,ha="right")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)

def graficar_evolucion_amortiguamientos(tiempos,C_hist,Cb_hist,ruta=None):
    t=_fechas(tiempos)
    C_hist=np.array(C_hist)
    Cb_hist=np.array(Cb_hist).ravel()
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for i in range(C_hist.shape[1]):
        axs[0].plot(t,C_hist[:,i],label="C_%d"%(i+1),marker=".",ms=3)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("C_i [N·s/m]")
    axs[0].legend(ncol=4)
    axs[0].set_title("Amortiguamiento de cojinetes")
    axs[1].plot(t,Cb_hist,marker=".",ms=3,color="k")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("C_b [N·s/m]")
    axs[1].set_xlabel("tiempo")
    axs[1].set_title("Amortiguamiento de banda")
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    axs[1].xaxis_date()
    plt.setp(axs[1].get_xticklabels(),rotation=30,ha="right")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)

def graficar_evolucion_modos(tiempos,w_hist,xi_hist,ruta=None):
    t=_fechas(tiempos)
    w_hist=np.array(w_hist)
    xi_hist=np.array(xi_hist)
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for i in range(w_hist.shape[1]):
        axs[0].plot(t,w_hist[:,i]/(2*np.pi),label="w_%d"%(i+1),marker=".",ms=3)
    axs[0].set_ylabel("frecuencia [Hz]")
    axs[0].legend(ncol=4)
    axs[0].set_title("Frecuencias modales")
    for i in range(xi_hist.shape[1]):
        axs[1].plot(t,xi_hist[:,i],label="xi_%d"%(i+1),marker=".",ms=3)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("xi_i")
    axs[1].set_xlabel("tiempo")
    axs[1].legend(ncol=4)
    axs[1].set_title("Factores de amortiguamiento")
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    axs[1].xaxis_date()
    plt.setp(axs[1].get_xticklabels(),rotation=30,ha="right")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)

def graficar_rmse_vs_tiempo(tiempos,rmse_hist,ruta=None):
    t=_fechas(tiempos)
    rmse_hist=np.array(rmse_hist)
    if rmse_hist.ndim==1:
        rmse_hist=rmse_hist[:,None]
    fig,ax=plt.subplots(figsize=(11,4))
    for j in range(rmse_hist.shape[1]):
        ax.plot(t,rmse_hist[:,j],label="canal %d"%(j+1),marker=".",ms=3)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("tiempo")
    ax.set_title("Error de pronostico por canal")
    ax.legend(ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.xaxis_date()
    plt.setp(ax.get_xticklabels(),rotation=30,ha="right")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)