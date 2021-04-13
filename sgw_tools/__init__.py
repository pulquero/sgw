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
import pygsp as gsp
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from scipy import sparse
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from builtins import staticmethod

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
        s = np.zeros((G.N, len(nodes)))
        for i, n in enumerate(nodes):
            s[n][i] = 1.0
    return s

def _tig(g, s, **kwargs):
    """
    Returns a tensor of (nodes, coeffs, filters)
    """
    tig = g.filter(s[..., np.newaxis], **kwargs)
    if s.shape[1] == 1: # single node
        tig = tig[:, np.newaxis, :]
    if tig.ndim == 2: # single filter
        tig = tig[..., np.newaxis]
    assert tig.shape == s.shape + (tig.shape[2],)
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

def spectrogram(G, g=None, M=100, nodes=None, **kwargs):
    if g is None:
        g = GaussianFilter(G, M)
    assert g.G == G
    g = ShiftedFilter(g, M)
    norm_sqr_func = lambda tig: np.linalg.norm(tig, axis=0, ord=2)**2
    spectr = nodeEmbedding(g, norm_sqr_func, nodes, **kwargs)
    G.spectr = spectr
    return spectr

def kernelCentrality(G, g=None, nodes=None, **kwargs):
    if g is None:
        g = gsp.filters.Heat(G)
    norm_sqr_func = lambda tig: np.linalg.norm(tig, axis=0, ord=2)**2
    centr = nodeEmbedding(g, norm_sqr_func, nodes, **kwargs)
    G.centr = centr
    return centr

def plotGraphWave(gw):
    if isinstance(gw, gsp.graphs.Graph):
        G = gw
        if not hasattr(G, 'gw'):
            graphWave(G)
        gw = G.gw

    fig, axs = plt.subplots(gw.shape[0], gw.shape[1], sharex='col', sharey='col')
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
    G.set_coordinates()
    G.plot()
    plotNodeLabels(G)

def plotSignal(G, y):
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

class LGraphFourier(gsp.graphs.fourier.GraphFourier):
    def compute_fourier_basis(self, recompute=False, spectrum_only=False):
        if hasattr(self, '_e') and hasattr(self, '_U') and not recompute:
            return

        assert self.L.shape == (self.N, self.N)
        if self.N > 3000:
            self.logger.warning('Computing the full eigendecomposition of a '
                                'large matrix ({0} x {0}) may take some '
                                'time.'.format(self.N))

        if spectrum_only:
            self._e = eigh(self.L.toarray(order='F'), eigvals_only=True, overwrite_a=True, driver='ev')
        else:
            self._e, self._U = eigh(self.L.toarray(order='F'), overwrite_a=True, driver='evr')
            self._mu = np.max(np.abs(self._U))

        self._e[np.isclose(self._e, 0)] = 0

        e_bound = self._get_upper_bound()
        self._e[np.isclose(self._e, e_bound)] = e_bound
        assert self._e[-1] <= e_bound

        assert np.max(self._e) == self._e[-1]
        self._lmax = self._e[-1]

    def _get_upper_bound(self):
        if self.lap_type == 'normalized' and hasattr(self, '_iw'):
            e_bound = np.max(self.iw)
            return e_bound
        elif self.lap_type == 'combinatorial':
            return np.inf
        else:
            raise Exception('Unsupported Laplacian type: {}'.format(self.lap_type))

class LGraph(LGraphFourier, gsp.graphs.Graph):
    def __init__(self, L, gtype='unknown', lap_type='combinatorial'):
        self.logger = gsp.utils.build_logger(__name__)
        self.L = L
        self.lap_type = lap_type
        self.N = L.shape[0]
        self.gtype = gtype

class Hypergraph(LGraph):
    def __init__(self, I, gtype='unknown', lap_type='combinatorial'):
        self.logger = gsp.utils.build_logger(__name__)
        self._W = None
        self._d = None
        self._dw = None
        self._i = None
        self._iw = None
        self.I = sparse.csr_matrix(I)
        self.lap_type = lap_type
        self.N = I.shape[0]
        self.Ne = I.shape[1]
        self.gtype = gtype
        self.compute_laplacian(lap_type)
        self.plotting = {'vertex_size': 100,
                         'vertex_color': (0.12, 0.47, 0.71, 1),
                         'edge_color': (0.5, 0.5, 0.5, 1),
                         'edge_width': 1,
                         'edge_style': '-'}

    def is_directed(self, recompute=False):
        return False

    def compute_laplacian(self, lap_type):
        self.L = self.I @ self.I.transpose()
        if lap_type == 'normalized':
            D_inv_sqrt = np.diagflat(1.0/np.sqrt(self.dw))
            self.L = D_inv_sqrt @ self.L @ D_inv_sqrt
        self.L = sparse.csc_matrix(self.L)

    @property
    def W(self):
        if self._W is None:
            w = (self.I @ self.I.transpose()).toarray()
            np.fill_diagonal(w, 0)
            self._W = sparse.lil_matrix(w)
        return self._W

    @property
    def d(self):
        if self._d is None:
            self._d = np.asarray(np.sum((self.I != 0).toarray(), axis=1)).squeeze()
        return self._d

    @property
    def dw(self):
        if self._dw is None:
            self._dw = np.asarray(np.sum(np.power(self.I.toarray(), 2), axis=1)).squeeze()
        return self._dw

    @property
    def i(self):
        if self._i is None:
            self._i = np.asarray(np.sum((self.I != 0).toarray(), axis=0)).squeeze()
        return self._i

    @property
    def iw(self):
        if self._iw is None:
            self._iw = np.asarray(np.sum(np.power(self.I.toarray(), 2), axis=0)).squeeze()
        return self._iw

class BigGraph(LGraphFourier, gsp.graphs.Graph):
    @staticmethod
    def create_from(G):
        coords = G.coords if hasattr(G, 'coords') else None
        return BigGraph(G.W, lap_type=G.lap_type, coords=coords, plotting=G.plotting)

    def __init__(self, W, lap_type='combinatorial', coords=None, plotting={}):
        gsp.graphs.Graph.__init__(self, W, lap_type=lap_type, coords=coords, plotting=plotting)

class BipartiteGraph(LGraphFourier, gsp.graphs.Graph):
    def __init__(self, W, lap_type='combinatorial', coords=None, plotting={}):
        gsp.graphs.Graph.__init__(self, W, lap_type=lap_type, coords=coords, plotting=plotting)
        self._lmax = 2

    def compute_fourier_basis(self, recompute=False):
        LGraphFourier.compute_fourier_basis(self, recompute)
        if self.lap_type == 'normalized':
            assert self.e[-1] == 2

    def _get_upper_bound(self):
        if self.lap_type == 'normalized':
            return 2
        elif self.lap_type == 'combinatorial':
            return 2*np.max(self.dw)
        else:
            raise Exception('Unsupported Laplacian type: {}'.format(self.lap_type))

class GWHeat(gsp.filters.Filter):
    """
    Heat kernel used by GraphWave.
    """
    def __init__(self, G, Nf=2, approximate=False):
        def kernel(x, s):
            return np.exp(-x * s)

        if approximate:
            if G.lap_type == 'normalized':
                lmin = 1.0/G.N
            else:
                lmin = np.sum(G.dw)/G.N/G.N
        else:
            lmin = G.e[np.invert(np.isclose(G.e, 0))][0]

        e_mean = np.sqrt(lmin * G.lmax)
        s_min = -np.log(0.95) / e_mean
        s_max = -np.log(0.80) / e_mean
        scales = np.linspace(s_min, s_max, Nf)
        kernels = [lambda x, s=s: kernel(x, s) for s in scales]

        gsp.filters.Filter.__init__(self, G, kernels)

class GaussianFilter(gsp.filters.Filter):
    """
    Filter used by spectrogram.
    """
    def __init__(self, G, M=100):
        def kernel(x):
            return np.exp(-M * (x/G.lmax)**2)

        gsp.filters.Filter.__init__(self, G, [kernel])

class ShiftedFilter(gsp.filters.Filter):
    """
    Filter used by spectrogram.
    """
    def __init__(self, g, Nf=100):
        G = g.G
        shifts = np.linspace(0, G.lmax, Nf)
        kernels = [lambda x, kernel=kernel, s=s: kernel(x - s) for s in shifts for kernel in g._kernels]

        gsp.filters.Filter.__init__(self, G, kernels)
