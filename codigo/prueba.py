import os,glob,numpy as np
from datetime import datetime

RUTA=r"C:\Users\chris\OneDrive\Documentos\GitHub\MangoLabs\proyecto_2\codigo\datos"
print("existe ruta:",os.path.isdir(RUTA))
archivos=glob.glob(os.path.join(RUTA,"*"))
print("n elementos en glob:",len(archivos))
for f in archivos[:5]:
    print("  ->",repr(os.path.basename(f)))
if archivos:
    f=archivos[0]
    print("es archivo:",os.path.isfile(f))
    try:
        arr=np.loadtxt(f)
        print("forma cargada:",arr.shape)
        print("dtype:",arr.dtype)
    except Exception as e:
        print("fallo loadtxt:",repr(e))