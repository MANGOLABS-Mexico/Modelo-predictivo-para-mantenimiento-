import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _fechas(t):
    return np.array(t,dtype="datetime64[s]")

def espectro_todos_canales(ymed,ypred_a,fs,ruta=None,titulo="Espectro por canal"):
    n=ymed.shape[0]
    f=np.fft.rfftfreq(n,1/fs)
    fig,axs=plt.subplots(4,2,figsize=(13,10),sharex=True)
    for c in range(8):
        ax=axs[c//2,c%2]
        Ym=np.abs(np.fft.rfft(ymed[:,c]))
        Yp=np.abs(np.fft.rfft(ypred_a[:,c]))
        ax.semilogy(f,Ym,label="medido",lw=0.8)
        ax.semilogy(f,Yp,label="analisis",lw=0.8,alpha=0.8)
        ax.set_title("canal %d"%(c+1),fontsize=9)
        ax.set_xlim(0,fs/2)
        ax.tick_params(labelsize=7)
        if c==0:
            ax.legend(fontsize=7)
    fig.suptitle(titulo)
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)

def aceleracion_todos_canales(t,ymed,ypred_a,ruta=None,titulo="Aceleracion por canal"):
    fig,axs=plt.subplots(4,2,figsize=(13,9),sharex=True)
    for c in range(8):
        ax=axs[c//2,c%2]
        ax.plot(t,ymed[:,c],lw=0.6,label="medido")
        ax.plot(t,ypred_a[:,c],lw=0.6,alpha=0.7,label="analisis")
        ax.set_title("canal %d"%(c+1),fontsize=9)
        ax.tick_params(labelsize=7)
        if c==0:
            ax.legend(fontsize=7)
    fig.suptitle(titulo)
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)

def residuo_por_canal_vs_tiempo(tiempos,res_hist,ruta=None,titulo="Residuo de analisis por canal"):
    t=_fechas(tiempos)
    R=np.array(res_hist)
    fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
    grupos=[(0,2,"cojinete 1 (canales 1,2)"),(2,4,"cojinete 2 (canales 3,4)"),
            (4,6,"cojinete 3 (canales 5,6)"),(6,8,"cojinete 4 (canales 7,8)")]
    for ax,(a,b,nombre) in zip(axs,grupos):
        for c in range(a,b):
            ax.plot(t,R[:,c],label="canal %d"%(c+1),marker=".",ms=2,lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("residuo")
        ax.set_title(nombre,fontsize=9)
        ax.legend(fontsize=7)
    axs[-1].set_xlabel("tiempo")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    axs[-1].xaxis_date()
    plt.setp(axs[-1].get_xticklabels(),rotation=30,ha="right")
    fig.suptitle(titulo)
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)

def residuo_heatmap(tiempos,res_hist,ruta=None,titulo="Residuo de analisis (canales x tiempo)"):
    t=_fechas(tiempos)
    R=np.array(res_hist).T
    fig,ax=plt.subplots(figsize=(13,4))
    im=ax.imshow(np.log10(R+1e-16),aspect="auto",origin="lower",cmap="viridis")
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels(["canal %d"%(i+1) for i in range(8)])
    n_cols=R.shape[1]
    idx=np.linspace(0,n_cols-1,min(8,n_cols)).astype(int)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(t[i])[:10] for i in idx],rotation=30,ha="right",fontsize=7)
    ax.set_xlabel("fecha")
    ax.set_title(titulo)
    cb=plt.colorbar(im,ax=ax,label="log10(residuo)")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.close(fig)