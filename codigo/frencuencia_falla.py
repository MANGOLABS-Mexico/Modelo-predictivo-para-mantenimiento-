import numpy as np
fr=33.3
Nb=16
d=8.4e-3
D=71.5e-3
alpha=np.deg2rad(15.17)
BPFO=Nb/2*fr*(1-d/D*np.cos(alpha))
BPFI=Nb/2*fr*(1+d/D*np.cos(alpha))
BSF=D/(2*d)*fr*(1-(d/D*np.cos(alpha))**2)