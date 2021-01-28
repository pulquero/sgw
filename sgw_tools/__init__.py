"""
0912.3848
1705.06250
1710.10321
1808.10650
Li_MSc_F2013 Spectral Geometric Methods for Deformable 3D Shape Retrieval
Ch1 Spectral Graph Theory
paperVC13 A multiresolution descriptor for deformable 3D shape retrieval
"""
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

"""
Refs: Shape Classification using Spectral Graph Wavelets/Spectral Geometric Methods for Deformable 3D Shape Retrieval

(12)/(3.14) is g.filter(f)
g(t, G.e[k]) is g.evaluate(G.e)[0, k]
(13)/(3.16) is psi_j(i) = g.localize(j)[i]/norm where norm is np.sqrt(G.N),
            which is just features.compute_tig(g) or g.compute_frame()
(19)/(3.21) is W_j(j) = g.localize(j)[j]/norm
"""

def heatKernel(G, t):
    def kernel(x):
        return np.exp(-t * x / G.lmax)
    return kernel

def gwt(f, G, kernel):
    f_hat = G.gft(f)
    coeffs = []
    for i in range(G.N):
        sum = 0.0
        for k in range(G.N):
            sum += kernel(G.e[k]) * f_hat[k] * G.U[i, k]
        coeffs.append(sum)
    return coeffs

def multiResolutionEmbedding(G, filter, R, ts, **kwargs):
    G.estimate_lmax()
    multiResEmbedding = []
    for L in range(1, R+1):
        g = filter(G, L)
        levelEmbedding = embedding(g, ts, **kwargs)
        multiResEmbedding.append(levelEmbedding)
    return np.concatenate(multiResEmbedding, axis=1)

"""
ts = np.linspace(0, 250, 25)
"""
def embedding(g, ts, nodes = None, **kwargs):
    if nodes is None:
        s = np.identity(g.G.N)
    else:
        s = np.zeros((g.G.N, len(nodes)))
        for i, n in enumerate(nodes):
            s[n][i] = 1.0
    tig = g.filter(s[..., np.newaxis], **kwargs)
    if tig.ndim == 2:
        tig = tig[..., np.newaxis]
    tig_t_grid = np.kron(tig[..., np.newaxis], ts)
    def chi(xt):
        return np.mean(np.exp(xt*1j), axis=0)
    return chi(tig_t_grid)
 
def plotEmbedding(embedding):
    fig, axs = plt.subplots(embedding.shape[0], embedding.shape[1], sharex='col', sharey='col')
    for n in range(embedding.shape[0]):
        for f in range(embedding.shape[1]):
            x = np.real(embedding[n][f])
            y = np.imag(embedding[n][f])
            if embedding.shape[1] > 1:
                axs[n][f].plot(x, y)
            else:
                axs[n].plot(x, y)
    fig.show()

def plotNodeLabels(G):
    for i, coord in enumerate(G.coords):
        plt.gca().annotate(str(i), coord)
    plt.gcf().show()
    plt.gcf().canvas.draw_idle()

def plotGraph(G):
    G.set_coordinates()
    G.plot()
    plotNodeLabels(G)

def plotSignal(G, y):
    G.set_coordinates()
    G.plot_signal(y)
    plotNodeLabels(G)
    
"""
Multi-resolution SGWs.
G = graph
filter(G, L) = kernel for G at level L, e.g. MexicanHat(G, Nf=L+1, normalize=True)
R = resolution (integer)
"""
def multiResolutionSignature(G, filter, R, **kwargs):
    G.estimate_lmax()
    multiResSig = []
    for L in range(1, R+1):
        g = filter(G, L)
        levelSig = signature(g, **kwargs)
        multiResSig.append(levelSig)
    return np.concatenate(multiResSig, axis=1)

"""
Exact implementation of SGW signature.
Requires calculation of Fourier basis.
"""
def _signature_exact(g):
    G = g.G
    G.compute_fourier_basis()
    ge = g.evaluate(G.e)
    sig = np.empty([ge.shape[1], ge.shape[0]])
    for i in range(ge.shape[1]):
        for t in range(ge.shape[0]):
            sum = 0.0
            for k in range(G.N):
                ev = G.U[i, k]
                sum += ge[t, k] * ev**2
            sig[i][t] = sum
    return sig
 
"""
Reference (PyGSP) implementation of SGW signature.
Approximate by default (avoids calculation of Fourier basis).
"""
def _signature_gsp(g, **kwargs):
    s = np.identity(g.G.N)
    tig = g.filter(s[..., np.newaxis], **kwargs)
    if tig.ndim == 2:
        tig = tig[..., np.newaxis]
    N = tig.shape[0] # graph size
    Nf = tig.shape[2] # num of filters
    sig = np.empty([N, Nf])
    for i in range(N):
        for t in range(Nf):
            sig[i][t] = tig[i][i][t]
    return sig

"""
SGW signature.
"""
signature = _signature_gsp

def codebook(sig, k):
    kmeans = cluster.KMeans(n_clusters=k, n_jobs=-1)
    kmeans.fit(sig)
    return _dedup(kmeans.cluster_centers_)
 
"""
Auto-tune the number of codewords (clusters).
""" 
def codebook_auto(sig, mink=2, maxk = None):
    if mink < 2:
        raise Exception('mink must be at least 2')
    if maxk is None:
        maxk = int(np.ceil(np.sqrt(sig.shape[0])))
    elif maxk < mink:
        raise Exception('mink cannot be greater than maxk')
    kmeans = _autoCluster(sig, mink, maxk)
    return _dedup(kmeans.cluster_centers_)

def _autoCluster(data, mink, maxk):
    maxScore = 0.0
    bestKMeans = None
    for k in range(mink, maxk+1):
        kmeans = cluster.KMeans(n_clusters=k, n_jobs=-1).fit(data)
        score = metrics.calinski_harabasz_score(data, kmeans.labels_)
        if score > maxScore:
            maxScore = score
            bestKMeans = kmeans
    return bestKMeans
    
def _dedup(arrs):
    if arrs.shape[0] > 1:
        prev = arrs[0]
        dedupped = []
        dedupped.append(prev)
        for k in range(1, arrs.shape[0]):
            next = arrs[k]
            if not np.allclose(next, prev):
                dedupped.append(next)
            prev = next
        return np.asarray(dedupped)
    else:
        return arrs

"""
SGW code.
alpha = 0 completely smoothed
"""
def code(sig, codebook, alpha):
    code = np.empty([codebook.shape[0], sig.shape[0]])
    def expDist(i, j):
        diff = sig[i] - codebook[j]
        return np.exp(-alpha*np.inner(diff, diff))
    for i in range(code.shape[1]):
        sum = 0.0
        for k in range(code.shape[0]):
            sum += expDist(i, k)
        for r in range(code.shape[0]):
            code[i][r] = expDist(i, r)/sum
    return code
    
"""
Histogram (code @ 1)
Sum across nodes for each feature.
Alternative aggregation functions include np.mean and np.amax.
"""
def histogram(code, agg=np.sum):
    N = code.shape[1] # graph size
    Nf = code.shape[0] # num of features
    h = np.empty([Nf])
    for i in range(Nf):
        h[i] = agg(code[i])
    return h
    
def bof(code, G, eps):
    K = np.exp(-G.W.toarray()/eps)
    return code @ K @ code.transpose()
 