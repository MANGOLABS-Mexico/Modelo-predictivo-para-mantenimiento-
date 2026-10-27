import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def graficar_prediccion_vs_medicion(t,ymedido,ypred_antes,ypred_despues,canal=0,ruta=None):
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
    axs[0].plot(t,ymedido[:,canal],label="medido",lw=1)
    axs[0].plot(t,ypred_antes[:,canal],label="fondo (pronostico)",lw=1,alpha=0.8)
    axs[0].set_ylabel("aceleracion canal %d"%canal)
    axs[0].legend()
    axs[0].set_title("Antes del analisis")
    axs[1].plot(t,ymedido[:,canal],label="medido",lw=1)
    axs[1].plot(t,ypred_despues[:,canal],label="analisis",lw=1,alpha=0.8)
    axs[1].set_ylabel("aceleracion canal %d"%canal)
    axs[1].set_xlabel("tiempo [s]")
    axs[1].legend()
    axs[1].set_title("Despues del analisis")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.show()
    plt.close(fig)

def graficar_error_por_canal(ymedido,ypred,fs,ruta=None):
    n_canales=ymedido.shape[1]
    rmse=np.sqrt(np.mean((ymedido-ypred)**2,axis=0))
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(np.arange(n_canales),rmse)
    ax.set_xlabel("canal")
    ax.set_ylabel("RMSE")
    ax.set_title("Error por canal")
    for i,v in enumerate(rmse):
        ax.text(i,v,"%.3g"%v,ha="center",va="bottom")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.show()
    plt.close(fig)
    return rmse

def graficar_espectro_comparado(ymedido,ypred,fs,canal=0,ruta=None):
    n=ymedido.shape[0]
    f=np.fft.rfftfreq(n,1/fs)
    Ym=np.abs(np.fft.rfft(ymedido[:,canal]))
    Yp=np.abs(np.fft.rfft(ypred[:,canal]))
    fig,ax=plt.subplots(figsize=(10,4))
    ax.semilogy(f,Ym,label="medido",lw=1)
    ax.semilogy(f,Yp,label="predicho",lw=1,alpha=0.8)
    ax.set_xlabel("frecuencia [Hz]")
    ax.set_ylabel("|F|")
    ax.set_xlim(0,fs/2)
    ax.legend()
    ax.set_title("Espectro canal %d"%canal)
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta,dpi=150)
    plt.show()
    plt.close(fig)

# Frecuencias características a 2000 RPM (Rexnord ZA-2115: D=2.815 in, d=0.331 in, 16 rodillos, 15.17 deg).
# Por geometría: BPFO=236.4, BPFI=296.9, BSF=139.9 Hz.
BPFO=236.4
BPFI=296.9
BSF=139.2

def _f(t):
    """Acepta fechas ISO ('2003-10-22T12:06:24') o nombres de archivo del IMS ('2003.10.22.12.06.24')."""
    out=[]
    for x in t:
        if isinstance(x,str) and len(x)==19 and x[4]=="." and x[7]==".":
            x=f"{x[0:4]}-{x[5:7]}-{x[8:10]}T{x[11:13]}:{x[14:16]}:{x[17:19]}"
        out.append(x)
    return np.array(out,dtype="datetime64[s]")

def _normalizar(X,n_base=10):
    """Divide por la mediana de los primeros n_base estimados (el primero solo es ruidoso)."""
    X=np.asarray(X,dtype=float)
    return X/np.median(X[:n_base],axis=0)

def _eje_fechas(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.xaxis_date()
    plt.setp(ax.get_xticklabels(),rotation=30,ha="right")

def graficar_R_normalizado(tiempos,R_hist,ruta=None,n_base=10):
    t=_f(tiempos)
    Rn=_normalizar(R_hist,n_base)
    fig,ax=plt.subplots(figsize=(11,5))
    for i,nom in enumerate(["R1 (coj.1)","R2 (coj.2)","R3 (coj.3)","R4 (coj.4)"]):
        ax.plot(t,Rn[:,i],label=nom,marker=".",ms=3)
    ax.axhline(0.5,ls="--",c="r",label="umbral -50%")
    ax.set_ylabel("R_i / R_i(sano)")
    ax.set_xlabel("tiempo")
    ax.set_title("Rigidez normalizada")
    ax.legend()
    _eje_fechas(ax)
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)

def graficar_C_normalizado(tiempos,C_hist,ruta=None,n_base=10):
    t=_f(tiempos)
    Cn=_normalizar(C_hist,n_base)
    fig,ax=plt.subplots(figsize=(11,5))
    for i,nom in enumerate(["C1","C2","C3 (coj.3)","C4 (coj.4)"]):
        ax.plot(t,Cn[:,i],label=nom,marker=".",ms=3)
    ax.set_yscale("log")
    ax.set_ylabel("C_i / C_i(sano)")
    ax.set_xlabel("tiempo")
    ax.set_title("Amortiguamiento normalizado")
    ax.legend()
    _eje_fechas(ax)
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)

def graficar_xi(tiempos,xi_hist,ruta=None):
    """xi_i(t) de cada modo, con la línea xi=1 (frontera subamortiguado / sobreamortiguado)."""
    t=_f(tiempos)
    xi=np.asarray(xi_hist,dtype=float)
    fig,ax=plt.subplots(figsize=(11,5))
    for i in range(xi.shape[1]):
        ax.plot(t,xi[:,i],label="modo %d"%(i+1),marker=".",ms=3)
    ax.axhline(1.0,ls="--",c="r",label="xi = 1")
    ax.set_yscale("log")
    ax.set_ylabel("factor de amortiguamiento xi")
    ax.set_xlabel("tiempo")
    ax.set_title("Régimen de amortiguamiento por modo")
    ax.legend()
    _eje_fechas(ax)
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)

def heatmap_espectro_canales(datos,tiempos,canales,ruta=None,titulo="",fmax=2000,fs=20000):
    """datos: lista de arreglos por archivo, (canales, muestras) o (muestras, canales);
    el eje de los canales es el más corto. Devuelve M[canal, archivo, frecuencia]."""
    n_arch=len(datos)
    def orientar(d):
        d=np.asarray(d)
        return d if d.shape[0]<d.shape[1] else d.T          # -> (canales, muestras)
    n_fft=orientar(datos[0]).shape[1]
    n_freq=int(fmax/(fs/n_fft))
    M=np.zeros((len(canales),n_arch,n_freq))
    for k in range(n_arch):
        y=orientar(datos[k])
        for j,c in enumerate(canales):
            Y=np.abs(np.fft.rfft(y[c]))
            M[j,k,:]=Y[:n_freq]
    fig,axs=plt.subplots(len(canales),1,figsize=(12,3*len(canales)),sharex=True)
    if len(canales)==1: axs=[axs]
    for j,c in enumerate(canales):
        ax=axs[j]
        im=ax.imshow(np.log10(M[j]+1e-12),aspect="auto",origin="lower",
                     extent=[0,fmax,0,n_arch],cmap="viridis")
        ax.axvline(BPFO,c="orange",ls="--",lw=0.8,label="BPFO")
        ax.axvline(BPFI,c="r",ls="--",lw=0.8,label="BPFI")
        ax.axvline(BSF,c="w",ls="--",lw=0.8,label="BSF")
        ax.axvline(2*BSF,c="w",ls=":",lw=0.8,label="2xBSF")
        ax.set_ylabel("archivo")
        ax.set_title("canal %d"%(c+1))
        ax.legend(fontsize=7,loc="upper right")
        plt.colorbar(im,ax=ax,label="log10|FFT|")
    axs[-1].set_xlabel("frecuencia [Hz]")
    fig.suptitle(titulo)
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)
    return M

def graficar_alarma(tiempos,R_hist,umbral=0.5,M=3,ruta=None,cojinete=2,n_base=10):
    """cojinete: índice 0..3 (por omisión 2 = cojinete 3). Devuelve el índice del archivo de la alarma o None."""
    t=_f(tiempos)
    Rn=_normalizar(R_hist,n_base)
    idx_alarma=None
    contador=0
    for i in range(len(Rn)):
        if Rn[i,cojinete]<umbral:
            contador+=1
            if contador>=M:
                idx_alarma=i
                break
        else:
            contador=0
    fig,ax=plt.subplots(figsize=(11,5))
    for i,nom in enumerate(["R1","R2","R3","R4"]):
        ax.plot(t,Rn[:,i],label=nom,marker=".",ms=3)
    if idx_alarma is not None:
        ax.axvline(t[idx_alarma],c="r",ls="--",lw=1.2,
                   label="ALARMA coj.%d: %s"%(cojinete+1,str(t[idx_alarma])[:10]))
        print("Alarma cojinete %d en"%(cojinete+1),t[idx_alarma])
    else:
        print("Sin alarma con umbral",umbral,"y M=",M)
    ax.axhline(umbral,ls=":",c="k",lw=0.8)
    ax.set_ylabel("R_i / R_i(sano)")
    ax.set_xlabel("tiempo")
    ax.set_title("Criterio de alarma - cojinete %d"%(cojinete+1))
    ax.legend()
    _eje_fechas(ax)
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)
    return idx_alarma

def graficar_w(tiempos,w_hist,ruta=None):
    """Frecuencia natural f_i = w_i/(2 pi) [Hz] de cada modo. w_hist: (n_archivos, 4) en rad/s."""
    t=_f(tiempos); f=np.asarray(w_hist,dtype=float)/(2*np.pi)
    fig,ax=plt.subplots(figsize=(11,5))
    for i in range(f.shape[1]):
        ax.plot(t,f[:,i],label="modo %d"%(i+1),marker=".",ms=3)
    ax.set_ylabel("frecuencia natural [Hz]"); ax.set_xlabel("tiempo")
    ax.set_title("Frecuencias naturales por modo"); ax.legend(); _eje_fechas(ax)
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)

def graficar_C_rayleigh(tiempos,aR_hist,bR_hist,w_hist,ruta=None,n_base=10):
    """Amortiguamiento de Rayleigh, C = a_R M + b_R K.
    Arriba: a_R y b_R respecto a su valor sano (mediana de los primeros n_base archivos).
    Abajo: coeficiente de amortiguamiento modal c_i = a_R + b_R w_i^2 = 2 xi_i w_i [1/s], por modo."""
    t=_f(tiempos); aR=np.asarray(aR_hist,dtype=float); bR=np.asarray(bR_hist,dtype=float)
    w=np.asarray(w_hist,dtype=float); c=aR[:,None]+bR[:,None]*w**2
    fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True)
    axs[0].plot(t,aR/np.median(aR[:n_base]),label="a_R / a_R(sano)",marker=".",ms=3)
    axs[0].plot(t,bR/np.median(bR[:n_base]),label="b_R / b_R(sano)",marker=".",ms=3)
    axs[0].set_yscale("log"); axs[0].set_ylabel("coeficiente / sano"); axs[0].legend()
    axs[0].set_title("Coeficientes de Rayleigh")
    for i in range(c.shape[1]):
        axs[1].plot(t,c[:,i],label="modo %d"%(i+1),marker=".",ms=3)
    axs[1].set_yscale("log"); axs[1].set_ylabel("c_i = 2 xi_i w_i  [1/s]"); axs[1].set_xlabel("tiempo")
    axs[1].set_title("Amortiguamiento modal por modo"); axs[1].legend(); _eje_fechas(axs[1])
    plt.tight_layout()
    if ruta: plt.savefig(ruta,dpi=150)
    plt.close(fig)