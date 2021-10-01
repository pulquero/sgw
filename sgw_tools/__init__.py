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


def kernelCentrality(G, g=None, nodes=None, **kwargs):
    if g is None:
        g = gsp.filters.Heat(G)
    norm_sqr_func = lambda tig: (np.linalg.norm(tig, axis=0, ord=2)**2)
    centr = nodeEmbedding(g, norm_sqr_func, nodes, **kwargs)
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
    G.set_coordinates()
    G.plot()
    plt.axis('off')
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


def magneticAdjacencyMatrix(G, q):
    if not G.is_directed():
        Wq = G.W
        dwq = G.dw
    else:
        W_star = G.W.T.conj()
        # hermitian
        W_h = (G.W + W_star)/2
        if q == 0:
            Wq = W_h
        else:
            # anti-hermitian
            W_anti = G.W - W_star
            gamma_data = np.exp(1j*2*np.pi*q*W_anti.data)
            Gamma = sparse.csr_matrix((gamma_data, W_anti.indices, W_anti.indptr), shape=W_anti.shape, dtype=complex)
            # Hadamar product
            Wq = Gamma.multiply(W_h)
        dwq = np.ravel(W_h.sum(axis=0))
    return Wq, dwq


def count_components(G):
    """
    Returns the number of weakly connected components.
    """
    W = G.Wq if hasattr(G, 'Wq') else magneticAdjacencyMatrix(G, q=0)
    unvisited = set(range(W.shape[0]))
    count = 0
    while unvisited:
        stack = [next(iter(unvisited))]
        while len(stack):
            v = stack.pop()
            if v in unvisited:
                unvisited.remove(v)
                stack.extend(W[v].nonzero()[1])
        count += 1
    return count


def extract_components(G):
    """
    Returns a list of the weakly connected components.
    """
    W = G.Wq if hasattr(G, 'Wq') else magneticAdjacencyMatrix(G, q=0)
    unvisited = set(range(W.shape[0]))
    subgraphs = []
    while unvisited:
        stack = [next(iter(unvisited))]
        comp = []
        while len(stack):
            v = stack.pop()
            if v in unvisited:
                unvisited.remove(v)
                comp.append(v)
                stack.extend(W[v].nonzero()[1])
        comp = sorted(comp)
        subG = G.subgraph(comp)
        subG.info = {'orig_idx': comp}
        subgraphs.append(subG)
    return subgraphs


def _estimate_lmin(G, maxiter):
    N = G.N
    approx_evecs = np.empty((N, 2))
    # 0-eigenvector (exact)
    approx_evecs[:,:-1] = 1
    # 1st non-zero eigenvector (guess)
    idxs = np.arange(N)
    one_idxs = np.random.choice(idxs, N//2, replace=False)
    neg_one_idxs = idxs[np.isin(idxs, one_idxs, assume_unique=True, invert=True)]
    approx_evecs[one_idxs,-1] = 1
    approx_evecs[neg_one_idxs,-1] = -1
    if N&1:
        approx_evecs[neg_one_idxs[-1],-1] = 0

    # simple pre-conditioner
    M = sparse.spdiags(1/G.L.diagonal(), 0, N, N)
    evals, _ = sparse.linalg.lobpcg(G.L, approx_evecs, M=M, largest=False, maxiter=maxiter)
    lmin = evals[-1]
    assert not np.isclose(lmin, 0), "Second eigenvalue is (close to) zero: {}".format(lmin)
    return lmin


def estimate_lmin(G, maxiter=2000):
    lmins = []
    for subG in G.extract_components():
        lmin = _estimate_lmin(subG, maxiter)
        lmins.append(lmin)
    return sorted(lmins)[0]


class LGraphFourier(gsp.graphs.fourier.GraphFourier):
    def compute_fourier_basis(self, recompute=False, spectrum_only=False):
        if hasattr(self, '_e') and self._e and hasattr(self, '_U') and self._U and not recompute:
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
        e_min = np.min(self._e)
        assert e_min == self._e[0], "First eigenvalue is not the smallest"
        assert e_min >= 0, "Smallest eigenvalue is negative {}".format(e_min)
        self._lmin = self._e[np.where(self._e>0)][0]

        e_bound = self._get_upper_bound()
        self._e[np.isclose(self._e, e_bound)] = e_bound
        e_max = np.max(self._e)
        assert e_max == self._e[-1], "Last eigenvalue is not the largest"
        assert e_max <= e_bound, "Largest eigenvalue was {} but upper bound is {}".format(e_max, e_bound)
        self._lmax = e_max

    def _get_upper_bound(self):
        return np.inf


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

    def _get_upper_bound(self):
        max_edge_degree = np.max(self.iw)
        if self.lap_type == 'normalized':
            return max_edge_degree
        elif self.lap_type == 'combinatorial':
            max_vertex_degree = np.max(self.dw)
            e_bound = max_edge_degree*max_vertex_degree
            return e_bound
        else:
            raise Exception('Unsupported Laplacian type: {}'.format(self.lap_type))

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
        """The degree (the number of incident nodes) of each edge."""
        if self._i is None:
            self._i = np.asarray(np.sum((self.I != 0).toarray(), axis=0)).squeeze()
        return self._i

    @property
    def iw(self):
        """The weighted degree of each edge."""
        if self._iw is None:
            self._iw = np.asarray(np.sum(np.power(self.I.toarray(), 2), axis=0)).squeeze()
        return self._iw


class BigGraph(LGraphFourier, gsp.graphs.Graph):
    @staticmethod
    def create_from(G):
        coords = G.coords if hasattr(G, 'coords') else None
        return BigGraph(G.W, lap_type=G.lap_type, coords=coords, plotting=G.plotting)

    def __init__(self, W, lap_type='combinatorial', q=0, coords=None, plotting={}):
        self.lap_type = None
        self._I = None
        self.q = q
        self._Iq = None
        self._Wq = None
        self._dwq = None
        self._lmin = None
        self._n_connected = None
        self._graph_init(W, lap_type=lap_type, coords=coords, plotting=plotting)

    def _graph_init(self, adjacency, lap_type='combinatorial', coords=None, plotting={}):

        self.logger = gsp.utils.build_logger(__name__)

        if not sparse.isspmatrix(adjacency):
            adjacency = np.asanyarray(adjacency)

        if (adjacency.ndim != 2) or (adjacency.shape[0] != adjacency.shape[1]):
            raise ValueError('Adjacency: must be a square matrix.')

        # CSR sparse matrices are the most efficient for matrix multiplication.
        # They are the sole sparse matrix type to support eliminate_zeros().
        self.W = sparse.csr_matrix(adjacency, copy=False)

        if np.isnan(self.W.sum()):
            raise ValueError('Adjacency: there is a Not a Number (NaN).')
        if np.isinf(self.W.sum()):
            raise ValueError('Adjacency: there is an infinite value.')
        if self.has_loops():
            self.logger.warning('Adjacency: there are self-loops '
                                '(non-zeros on the diagonal). '
                                'The Laplacian will not see them.')
        if (self.W < 0).nnz != 0:
            self.logger.warning('Adjacency: there are negative edge weights.')

        self.gtype = 'unknown'

        self.N = self.W.shape[0]

        # Don't keep edges of 0 weight. Otherwise Ne will not correspond
        # to the real number of edges. Problematic when plotting.
        self.W.eliminate_zeros()

        # Don't count edges two times if undirected.
        # Be consistent with the size of the differential operator.
        if self.is_directed():
            self.Ne = self.W.nnz
        else:
            diagonal = np.count_nonzero(self.W.diagonal())
            off_diagonal = self.W.nnz - diagonal
            self.Ne = off_diagonal // 2 + diagonal

        self.compute_laplacian(lap_type)

        if coords is not None:
            # TODO: self.coords should be None if unset.
            self.coords = np.asanyarray(coords)

        self.plotting = {
                'vertex_size': 100,
                'vertex_color': (0.12, 0.47, 0.71, 1),
                'edge_color': (0.5, 0.5, 0.5, 1),
                'edge_width': 1,
                'edge_style': '-'
        }
        self.plotting.update(plotting)

    def _get_upper_bound(self):
        if self.lap_type == 'normalized':
            return 2
        elif self.lap_type == 'combinatorial':
            return 2*np.max(self.dw)
        else:
            raise Exception('Unsupported Laplacian type: {}'.format(self.lap_type))

    def has_loops(self):
        return np.any(self.W.diagonal() != 0)

    def compute_laplacian(self, lap_type='combinatorial'):
        if lap_type != self.lap_type:
            # Those attributes are invalidated when the Laplacian is changed.
            # Alternative: don't allow the user to change the Laplacian.
            self._lmin = None
            if hasattr(self, '_lmax'):
                del self._lmax
            if hasattr(self, '_U'):
                del self._U
            if hasattr(self, '_e'):
                del self._e
            if hasattr(self, '_coherence'):
                del self._coherence
            if hasattr(self, '_D'):
                del self._D

        self.lap_type = lap_type

        if lap_type == 'combinatorial':
            D = sparse.diags(self.dwq)
            self.L = D - self.Wq
        elif lap_type == 'normalized':
            d = np.zeros(self.N)
            disconnected = (self.dwq == 0)
            np.power(self.dwq, -0.5, where=~disconnected, out=d)
            D = sparse.diags(d)
            self.L = sparse.identity(self.N) - D * self.Wq * D
            self.L[disconnected, disconnected] = 0
            self.L.eliminate_zeros()
        else:
            raise ValueError('Unknown Laplacian type {}'.format(lap_type))

    def get_edge_list(self):
        if self.is_directed():
            W = self.W.tocoo()
        else:
            W = sparse.triu(self.W, format='coo')

        sources = W.row
        targets = W.col
        weights = W.data

        assert self.Ne == sources.size == targets.size == weights.size
        return sources, targets, weights

    def subgraph(self, vertices):
        adjacency = self.W[vertices, :][:, vertices]
        try:
            coords = self.coords[vertices]
        except AttributeError:
            coords = None
        return BigGraph(adjacency, self.lap_type, self.q, coords, self.plotting)

    @property
    def I(self):
        if self._I is None:
            self._I = sparse.lil_matrix((self.N, self.Ne))
            e = 0
            for i, row in enumerate(self.W.rows):
                for offset, j in enumerate(row):
                    if self.is_directed() or i > j:
                        v = self.W.data[i][offset]
                        v_rt = np.sqrt(v)
                        self._I[j,e] = v_rt
                        self._I[i,e] = -v_rt
                        e += 1
            self._I = self._I.tocsr()
        return self._I

    @property
    def i(self):
        """The degree (the number of incident nodes) of each edge."""
        return 2

    @property
    def Iq(self):
        if self._Iq is None:
            self._Iq = sparse.lil_matrix((self.N, self.Ne), dtype=complex)
            e = 0
            for i,j,v in zip(*sparse.find(self.Wq)):
                if i > j:
                    v_rt = np.sqrt(v)
                    self._Iq[i,e] = v_rt
                    self._Iq[j,e] = -v_rt.conj()
                    e += 1
            self._Iq = self._Iq.tocsr()
        return self._Iq

    @property
    def Wq(self):
        if self._Wq is None:
            self._Wq, self._dwq = magneticAdjacencyMatrix(self, self.q)
        return self._Wq

    @property
    def dwq(self):
        if self._dwq is None:
            self._Wq, self._dwq = magneticAdjacencyMatrix(self, self.q)
        return self._dwq

    @property
    def lmin(self):
        if self._lmin is None:
            self.logger.warning('The smallest non-zero eigenvalue G.lmin is not '
                                'available, we need to estimate it. '
                                'Explicitly call G.estimate_lmin() or '
                                'G.compute_fourier_basis() '
                                'once beforehand to suppress this warning.')
            self.estimate_lmin()
        return self._lmin

    def estimate_lmin(self):
        if self._lmin is None:
            self._lmin = estimate_lmin(self)

    def count_components(self):
        if self._n_connected is None:
            self._n_connected = count_components(self)
            if not hasattr(self, '_connected'):
                self._connected = (self._n_connected == 1)
        return self._n_connected

    def extract_components(self):
        return extract_components(self)


class BipartiteGraph(BigGraph):
    def __init__(self, W, lap_type='combinatorial', q=0, coords=None, plotting={}):
        super().__init__(W, lap_type=lap_type, q=q, coords=coords, plotting=plotting)
        self._lmax = 2

    def compute_fourier_basis(self, recompute=False):
        LGraphFourier.compute_fourier_basis(self, recompute)
        if self.lap_type == 'normalized':
            assert self.e[-1] == 2


class StarGraph(gsp.graphs.Comet):
    def __init__(self, N):
        super().__init__(N, N-1)


class DirectedPath(BigGraph):
    def __init__(self, N, **kwargs):
        sources = np.arange(0, N-1)
        targets = np.arange(1, N)
        weights = np.ones(N-1)
        W = sparse.csr_matrix((weights, (sources, targets)), shape=(N, N))
        super().__init__(W, **kwargs)


class DirectedRing(DirectedPath):
    def __init__(self, N, **kwargs):
        super().__init__(N, **kwargs)
        self.W[N-1,0] = 1


class GWHeat(gsp.filters.Filter):
    """
    Heat kernel used by GraphWave.
    normalize helps maintain a consistent sampling of the ECF across different filter scales,
        but at the cost of calculating all the eigenvalues.
    """
    def __init__(self, G, Nf=2, normalize=False, gamma=0.95, eta=0.85, approximate=False, maxiter=2000):
        def kernel(x, s):
            return np.exp(-x * s)

        if hasattr(G, '_lmin') and G._lmin:
            lmin = G._lmin
        else:
            if not approximate and G.N > 3000:
                G.logger.warning('Large matrix ({0} x {0}) detected - using faster approximation'.format(G.N))
                approximate = True
            if approximate:
                lmin = estimate_lmin(G, maxiter=maxiter)
            else:
                lmin = G.e[np.invert(np.isclose(G.e, 0))][0]
            G._lmin = lmin

        e_mean = np.sqrt(lmin * G.lmax)
        s_min = -np.log(gamma) / e_mean
        s_max = -np.log(eta) / e_mean
        assert s_min < s_max, 's_min ({}) is not less than s_max ({})'.format(s_min, s_max)
        # log scale
        self.scales = np.exp(np.linspace(np.log(s_min), np.log(s_max), Nf))
        kernels = []
        for s in self.scales:
            if normalize:
                norm = np.linalg.norm(kernel(G.e, s))
                kernels.append(lambda x, s=s, norm=norm: kernel(x, s)/norm)
            else:
                kernels.append(lambda x, s=s: kernel(x, s))

        super().__init__(G, kernels)


class GaussianFilter(gsp.filters.Filter):
    """
    Filter used by spectrogram.
    """
    def __init__(self, G, M=100):
        def kernel(x):
            return np.exp(-M * (x/G.lmax)**2)

        super().__init__(G, [kernel])


class ShiftedFilter(gsp.filters.Filter):
    """
    Filter used by spectrogram.
    """
    def __init__(self, g, Nf=100, shifts=None):
        G = g.G

        if shifts is None:
            shifts = np.linspace(0, G.lmax, Nf)

        if len(shifts) != Nf:
            raise ValueError('len(shifts) should be Nf.')

        kernels = [lambda x, kernel=kernel, s=s: kernel(x - s) for s in shifts for kernel in g._kernels]

        super().__init__(G, kernels)


class CustomFilter(gsp.filters.Filter):
    def __init__(self, G, func, scales=1):
        def kernel(x, s):
            return func(x*s)

        if not hasattr(scales, '__iter__'):
            scales = [scales]
        self.scales = scales

        kernels = []
        for s in scales:
            kernels.append(lambda x, s=s: kernel(x,s))
        super().__init__(G, kernels)
