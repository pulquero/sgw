import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw

G = gsp.graphs.Comet(20, 11)
G.estimate_lmax()

g = gsp.filters.Heat(G)
data = sgw.kernelCentrality(G, g)

nodes = np.arange(G.N)
ranking = sorted(nodes, key=lambda v: 1/data[v], reverse=True)

sgw.plotGraph(G)
print(ranking)
