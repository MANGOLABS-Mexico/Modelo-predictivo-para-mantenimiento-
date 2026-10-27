import numpy as np
import os
import glob
import re
from datetime import datetime

RUTA_DATOS=r"C:\Users\chris\OneDrive\Documentos\GitHub\MangoLabs\proyecto_2\codigo\datos"
FS=20000
N_MUESTRAS=20480
N_CANALES=8
PATRON=re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})")

def _parsear_fecha(nombre):
    m=PATRON.search(nombre)
    if not m:
        return None
    try:
        return datetime(*[int(g) for g in m.groups()])
    except ValueError:
        return None

def _leer_archivo(f):
    try:
        arr=np.loadtxt(f,encoding="latin-1")
    except Exception:
        try:
            arr=np.genfromtxt(f,encoding="latin-1")
        except Exception:
            return None
    if arr.ndim!=2:
        return None
    if arr.shape==(N_CANALES,N_MUESTRAS):
        return arr
    if arr.shape==(N_MUESTRAS,N_CANALES):
        return arr.T
    return None

def listar_archivos_conjunto1(ruta=RUTA_DATOS,verbose=True):
    if not os.path.isdir(ruta):
        if verbose:
            print("[listar] la ruta no existe:",ruta)
        return []
    candidatos=[]
    for f in glob.glob(os.path.join(ruta,"*")):
        if not os.path.isfile(f):
            continue
        nombre=os.path.basename(f)
        t=_parsear_fecha(nombre)
        if t is None:
            continue
        candidatos.append((t,f))
    if verbose:
        print("[listar] candidatos con nombre de fecha:",len(candidatos))
    validos=[]
    for t,f in sorted(candidatos,key=lambda p:p[0]):
        arr=_leer_archivo(f)
        if arr is None:
            continue
        validos.append((t,f))
    if verbose:
        print("[listar] validos:",len(validos))
    return validos

def cargar_archivo(ruta):
    arr=_leer_archivo(ruta)
    t=_parsear_fecha(os.path.basename(ruta))
    return arr,t

def cargar_todo_conjunto1(ruta=RUTA_DATOS,verbose=True):
    pares=listar_archivos_conjunto1(ruta,verbose=verbose)
    if len(pares)==0:
        raise RuntimeError("No se encontro ningun archivo valido en %s"%ruta)
    datos=[]
    tiempos=[]
    for t,f in pares:
        arr=_leer_archivo(f)
        if arr is None:
            continue
        datos.append(arr)
        tiempos.append(t)
    datos=np.array(datos)
    tiempos=np.array(tiempos)
    if verbose:
        print("[cargar] datos:",datos.shape,"desde",tiempos[0],"hasta",tiempos[-1])
    return datos,tiempos