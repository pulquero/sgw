"""
0912.3848 Wavelets on Graphs via Spectral Graph Theory
1705.06250 Shape Classification using Spectral Graph Wavelets
1710.10321 Learning Structural Node Embeddings Via Diffusion Wavelets
1808.10650 Graph reduction with spectral and cut guarantees
Li_MSc_F2013 Spectral Geometric Methods for Deformable 3D Shape Retrieval
Ch1 Spectral Graph Theory
paperVC13 A multiresolution descriptor for deformable 3D shape retrieval
"""
import numpy as np
import pygsp as gsp
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from scipy import sparse
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from builtins import staticmethod

from .graph import LGraph
from .graph import Hypergraph
from .graph import BigGraph
from .graph import BipartiteGraph
from .graphs import RingGraph
from .graphs import StarGraph
from .graphs import DirectedPath
from .graphs import DirectedRing
from .graphs import RandomRegular
from .filters import GWHeat
from .filters import GaussianFilter
from .filters import ShiftedFilter
from .filters import ChebyshevFilter
from .filters import CustomFilter


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
        total = 0.0
        for k in range(G.N):
            total += kernel(G.e[k]) * f_hat[k] * G.U[i, k]
        coeffs.append(total)
    return coeffs


def createSignal(G, nodes=None):
    if nodes is None:
        s = np.identity(G.N)
    else:
        if not hasattr(nodes, '__iter__'):
            nodes = [nodes]

        s = np.zeros((G.N, len(nodes)))
        for i, n in enumerate(nodes):
            s[n][i] = 1.0
    return s


def _tig(g, s, **kwargs):
    """
    Returns a tensor of (coeffs, nodes, filters)
    """
    if g.G.is_directed() and g.G.q != 0 and ('method' not in kwargs or kwargs['method'] != 'exact'):
        raise Exception("Only method='exact' is currently supported for magnetic Laplacians.")

    if s.ndim == 1:  # single signal
        s = s[..., np.newaxis]
    assert s.ndim == 2, "Signal shape was {}".format(s.shape)

    if g.G.N == 1:
        kwargs['method'] = 'exact'
    tig = g.filter(s[..., np.newaxis], **kwargs)

    if tig.ndim == 0:
        tig = tig[np.newaxis, np.newaxis, np.newaxis]
    elif tig.ndim == 1:
        if s.shape == (1,1):  # multiple filters
            tig = tig[np.newaxis, np.newaxis, ...]
        elif s.shape[0] == 1:  # single node
            tig = tig[np.newaxis, ..., np.newaxis]
        elif s.shape[1] == 1:  # single signal
            tig = tig[:, np.newaxis, np.newaxis]
    elif tig.ndim == 2:
        if s.shape[0] == 1:  # single node
            tig = tig[np.newaxis, ...]
        elif s.shape[1] == 1:  # single signal
            tig = tig[:, np.newaxis, :]
        else:  # single filter
            tig = tig[..., np.newaxis]

    assert tig.shape == s.shape + (tig.shape[2],), "Tig shape was {}".format(tig.shape)
    return tig


def nodeEmbedding(g, func, nodes=None, **kwargs):
    s = createSignal(g.G, nodes)
    tig = _tig(g, s, **kwargs)
    return func(tig)


def multiResolutionGraphWave(G, filter_factory, R, ts, **kwargs):
    """
    Multi-resolution GraphWave.
    G = graph
    filter_factory(G, L) = kernel for G at level L, e.g. MexicanHat(G, Nf=L+1, normalize=True)
    R = resolution (integer for levels 1..R else iterator of levels)
    ts = np.linspace(0, 100, 25)
    """
    if type(R) == int:
        R = range(1, R+1)
    G.estimate_lmax()
    multiResGW = []
    for L in R:
        g = filter_factory(G, L)
        gw = graphWave(G, g, ts, **kwargs)
        multiResGW.append(gw)
    return np.concatenate(multiResGW, axis=1)


def graphWave(G, g=None, ts=None, nodes=None, ecf_method=None, **kwargs):
    """
    GraphWave.
    ts = np.linspace(0, 100, 25)
    ecf_method = method to calculate the empirical characteristic function
    Returns a tensor of (nodes, filters, chi)
    """
    if g is None:
        g = GWHeat(G)
    assert g.G == G
    if ts is None:
        ts = np.linspace(0, 100, 25)

    def char_func(tig, ecf_method):
        N = tig.shape[0]
        if ecf_method is None:
            if N < 100:
                ecf_method = 'kron'
            elif N < 10000:
                ecf_method = 'partial-kron'
            else:
                ecf_method = 'loop'
    
        if ecf_method == 'kron':
            # fully vectorized
            tig_t_grid = np.kron(tig[..., np.newaxis], ts)
            def chi(xt):
                return np.mean(np.exp(xt*1j), axis=0)
            return chi(tig_t_grid)
        elif ecf_method == 'partial-kron':
            # more memory efficient
            def chi(xt):
                return np.mean(np.exp(xt*1j), axis=0)
            gw = np.empty((tig.shape[1], tig.shape[2], ts.shape[0]), dtype=complex)
            for i in range(tig.shape[1]):
                for j in range(tig.shape[2]):
                    tig_t_grid = np.kron(tig[:,i,j, np.newaxis], ts)
                    gw[i][j] = chi(tig_t_grid)
            return gw
        elif ecf_method == 'loop':
            # every byte counts
            def chi(x, t):
                return np.mean(np.exp(x*t*1j), axis=0)
            gw = np.empty((tig.shape[1], tig.shape[2], ts.shape[0]), dtype=complex)
            for i in range(tig.shape[1]):
                for j in range(tig.shape[2]):
                    for k, t in enumerate(ts):
                        gw[i][j][k] = chi(tig[:,i,j], t)
            return gw

    gw = nodeEmbedding(g, lambda tig: char_func(tig, ecf_method), nodes, **kwargs)
    G.gw = gw
    return gw


def gw_flatten(gw):
    data_z = gw.reshape(gw.shape[0], gw.shape[1]*gw.shape[2])
    return np.c_[data_z.real, data_z.imag]


def spectrogram(G, g=None, M=100, shifts=None, nodes=None, **kwargs):
    if g is None:
        g = GaussianFilter(G, M)
    assert g.G == G

    if shifts is None:
        Nf = M
    else:
        Nf = len(shifts)
    g = ShiftedFilter(g, Nf, shifts)

    norm_sqr_func = lambda tig: np.linalg.norm(tig, axis=0, ord=2)**2
    spectr = nodeEmbedding(g, norm_sqr_func, nodes, **kwargs)
    G.spectr = spectr
    return spectr


def kernelCentrality(G, g=None, nodes=None, ord=None, **kwargs):
    if g is None:
        g = gsp.filters.Heat(G)
    norm_func = lambda tig: np.linalg.norm(tig, axis=0, ord=ord)
    centr = nodeEmbedding(g, norm_func, nodes, **kwargs)
    G.centr = centr
    return centr


def plotTig(tig, nodes=None):
    if isinstance(tig, gsp.filters.Filter):
        g = tig
        tig = _tig(g, createSignal(g.G))
    else:
        g = None

    if nodes is None:
        nodes = range(tig.shape[0])
    elif not hasattr(nodes, '__iter__'):
        nodes = [nodes]

    num_nodes = len(nodes)
    num_filters = tig.shape[2]
    fig, axs = plt.subplots(num_nodes, num_filters, num='Wavelet coefficients', sharex='col', sharey='col')
    title = '{} wavelet coefficients'.format(g.__class__.__name__) if g else 'Wavelet coefficients'
    fig.suptitle(title)
    for i, n in enumerate(nodes):
        for f in range(num_filters):
            if num_nodes > 1 and num_filters > 1:
                axs[i][f].plot(tig[n,:,f])
            elif num_nodes == 1 and num_filters > 1:
                axs[f].plot(tig[n,:,f])
            elif num_nodes > 1 and num_filters == 1:
                axs[i].plot(tig[n,:,f])
            else:
                axs.plot(tig[n,:,f])
    fig.show()
    return fig


def plotGraphWave(gw):
    if isinstance(gw, gsp.graphs.Graph):
        G = gw
        if not hasattr(G, 'gw'):
            graphWave(G)
        gw = G.gw

    fig, axs = plt.subplots(gw.shape[0], gw.shape[1], num='ECF', sharex='col', sharey='col')
    fig.suptitle('ECF')
    for n in range(gw.shape[0]):
        for f in range(gw.shape[1]):
            x = np.real(gw[n][f])
            y = np.imag(gw[n][f])
            if gw.shape[1] > 1:
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
    if not hasattr(G, 'coords') or G.coords is None:
        G.set_coordinates()
    G.plot()
    plt.axis('off')
    plotNodeLabels(G)


def plotSignal(G, y):
    if not hasattr(G, 'coords') or G.coords is None:
        G.set_coordinates()
    G.plot_signal(y)
    plotNodeLabels(G)


def multiResolutionSignature(G, filter_factory, R, **kwargs):
    """
    Multi-resolution SGWs.
    G = graph
    filter_factory(G, L) = kernel for G at level L, e.g. MexicanHat(G, Nf=L+1, normalize=True)
    R = resolution (integer for levels 1..R else iterator of levels)
    """
    if type(R) == int:
        R = range(1, R+1)
    G.estimate_lmax()
    multiResSig = []
    for L in R:
        g = filter_factory(G, L)
        levelSig = signature(g, **kwargs)
        multiResSig.append(levelSig)
    return np.concatenate(multiResSig, axis=1)


def _signature_exact(g):
    """
    Exact implementation of SGW signature.
    Requires calculation of Fourier basis.
    """
    G = g.G
    G.compute_fourier_basis()
    ge = g.evaluate(G.e)
    sig = np.empty([ge.shape[1], ge.shape[0]])
    for i in range(ge.shape[1]):
        for t in range(ge.shape[0]):
            total = 0.0
            for k in range(G.N):
                ev = G.U[i, k]
                total += ge[t, k] * ev**2
            sig[i][t] = total
    return sig


def _signature_gsp(g, **kwargs):
    """
    Reference (PyGSP) implementation of SGW signature.
    Approximate by default (avoids calculation of Fourier basis).
    """
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
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(sig)
    return _dedup(kmeans.cluster_centers_)


def codebook_auto(sig, mink=2, maxk = None):
    """
    Auto-tune the number of codewords (clusters).
    """ 
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
        kmeans = cluster.KMeans(n_clusters=k).fit(data)
        score = metrics.calinski_harabasz_score(data, kmeans.labels_)
        if score > maxScore:
            maxScore = score
            bestKMeans = kmeans
    return bestKMeans
    

def _dedup(arrs):
    if arrs.shape[0] > 1:
        _prev = arrs[0]
        dedupped = []
        dedupped.append(_prev)
        for k in range(1, arrs.shape[0]):
            _next = arrs[k]
            if not np.allclose(_next, _prev):
                dedupped.append(_next)
            _prev = _next
        return np.asarray(dedupped)
    else:
        return arrs


def code(sig, codebook, alpha):
    """
    SGW code.
    alpha = 0 completely smoothed
    """
    code = np.empty([codebook.shape[0], sig.shape[0]])
    def expDist(i, j):
        diff = sig[i] - codebook[j]
        return np.exp(-alpha*np.inner(diff, diff))
    for r in range(code.shape[0]):
        for i in range(code.shape[1]):
            code[r][i] = expDist(i, r)
    code /= np.sum(code, axis=0)
    return code


def histogram(code, agg=np.sum):
    """
    Histogram (code @ 1)
    Sum across nodes for each feature.
    Alternative aggregation functions include np.mean and np.amax.
    """
    N = code.shape[1] # graph size
    Nf = code.shape[0] # num of features
    h = np.empty([Nf])
    for i in range(Nf):
        h[i] = agg(code[i])
    return h


def bof(code, G, eps):
    K = np.exp(-G.W.toarray()/eps)
    return code @ K @ code.transpose()

