import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw

Ip = np.array([[1,0,0],[-1,1,0],[0,-1,1],[0,0,-1]])
Ipw = np.array([[np.sqrt(2),0,0],[-np.sqrt(2),1,0],[0,-1,np.sqrt(3)],[0,0,-np.sqrt(3)]])
I3 = np.array([[1,1,1,0,0],[0,1,1,1,0],[1,0,1,1,0],[0,1,0,1,1]]).transpose()
I3w = np.array([[np.sqrt(2),np.sqrt(2),np.sqrt(2),0,0],[0,1,1,1,0],[np.sqrt(3),0,np.sqrt(3),np.sqrt(3),0],[0,1,0,1,1]]).transpose()
Imixed = np.array([[1,np.sqrt(2),0,0,1],[1,0,np.sqrt(2),1,0],[-1,0,0,-np.sqrt(2),1],[0,-np.sqrt(2),0,-1,1],[0,0,0,0,-np.sqrt(3)]])
Is = [Ip, Ipw, I3, I3w, Imixed]
for I in Is:
    for lap_type in ['combinatorial', 'normalized']:
        G = sgw.Hypergraph(I, lap_type=lap_type)
        G.compute_fourier_basis()
        print(G.e)
