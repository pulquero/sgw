import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw
import matplotlib.pyplot as plt

G = gsp.graphs.Comet(20, 11)
G.estimate_lmax()

g = gsp.filters.Heat(G)
data = sgw.kernelCentrality(G, g)

nodes = np.arange(G.N)
ranking = sorted(nodes, key=lambda v: 1/data[v][0], reverse=True)

sgw.plotGraph(G)
print(["{} ({:.3f})".format(r,data[r][0]) for r in ranking])
plt.show()
