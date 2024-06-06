import numpy as np
import pygsp as gsp
from scipy import sparse
from scipy.linalg import eigh
from . import util


class LGraphFourier(gsp.graphs.fourier.GraphFourier):
    def _has_fourier_basis(self, recompute):
        if hasattr(self, '_e') and self._e is not None and hasattr(self, '_U') and self._U is not None and not recompute:
            return True

    def compute_fourier_basis(self, recompute=False, spectrum_only=False):
        if self._has_fourier_basis(recompute):
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
        e_nz = self._e[np.where(self._e>0)]
        if len(e_nz) > 0:
            self._lmin = e_nz[0]
        else:
            self._lmin = np.nan

        e_bound = self._get_upper_bound()
        self._e[np.isclose(self._e, e_bound)] = e_bound
        e_max = np.max(self._e)
        assert e_max == self._e[-1], "Last eigenvalue is not the largest"
        assert e_max <= e_bound, "Largest eigenvalue was {} but upper bound is {}".format(e_max, e_bound)
        self._lmax = e_max

    def _get_upper_bound(self):
        return np.inf

    @property
    def coherence(self):
        r"""Coherence of the Fourier basis.
        Is computed by :func:`compute_fourier_basis`.
        """
        return self._check_fourier_properties('mu', 'Fourier basis coherence')


class LGraph(LGraphFourier, gsp.graphs.Graph):
    def __init__(self, L, gtype='unknown', lap_type='combinatorial'):
        self.logger = gsp.utils.build_logger(__name__)
        self.L = L
        self.lap_type = lap_type
        self.n_vertices = self.N = L.shape[0]
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
        self.n_vertices = self.N = I.shape[0]
        self.n_edges = self.Ne = I.shape[1]
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
            return np.inf

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
    def create_from(G, lap_type=None, s=1, q=0):
        if lap_type is None:
            lap_type = G.lap_type
        coords = G.coords if hasattr(G, 'coords') else None
        return BigGraph(G.W, lap_type=lap_type, s=s, q=q, coords=coords, plotting=G.plotting)

    def __init__(self, W, lap_type='combinatorial', s=1, q=0, coords=None, plotting={}):
        """
        s = deformation parameter (1 - sA + s^2(D - 1))
        q = magnetic parameter [0, 1]
        """
        self.lap_type = None
        self._I = None
        self.s = s
        self.q = q
        self._Iq = None
        self._Wq = None
        self._d = None
        self._dw = None
        self._dwq = None
        self._lmin = None
        self._lmax = None
        self._lmax_method = None
        self._n_connected = None
        self._graph_init(W, lap_type=lap_type, coords=coords, plotting=plotting)

    def _graph_init(self, adjacency, lap_type='combinatorial', coords=None, plotting={}):

        self.logger = gsp.utils.build_logger(__name__)

        if not sparse.issparse(adjacency):
            # if adjacency is of a specialist/read-only type (e.g. when using ray)
            adjacency = np.asanyarray(adjacency)

        if (adjacency.ndim != 2) or (adjacency.shape[0] != adjacency.shape[1]):
            raise ValueError('Adjacency: must be a square matrix.')

        # CSR sparse matrices are the most efficient for matrix multiplication.
        # They are the sole sparse matrix type to support eliminate_zeros().
        self.W = sparse.csr_matrix(adjacency, copy=False)

        element_sum = self.W.sum()
        if np.isnan(element_sum):
            raise ValueError('Adjacency: there is a Not a Number (NaN).')
        if np.isinf(element_sum):
            raise ValueError('Adjacency: there is an infinite value.')
        if self.has_loops():
            self.logger.warning('Adjacency: there are self-loops '
                                '(non-zeros on the diagonal). '
                                'The Laplacian will not see them.')
        if util.count_negatives(self.W) != 0:
            self.logger.warning('Adjacency: there are negative edge weights.')

        self.gtype = 'unknown'

        self.n_vertices = self.N = self.W.shape[0]

        # Don't keep edges of 0 weight. Otherwise Ne will not correspond
        # to the real number of edges. Problematic when plotting.
        if self.W.nnz > self.W.count_nonzero():  # avoid unnecessary mutations in case underlying matrix is read-only
            self.W.eliminate_zeros()

        # Don't count edges two times if undirected.
        # Be consistent with the size of the differential operator.
        if self.is_directed():
            self.n_edges = self.Ne = self.W.nnz
        else:
            diagonal = np.count_nonzero(self.W.diagonal())
            off_diagonal = self.W.nnz - diagonal
            self.n_edges = self.Ne = off_diagonal // 2 + diagonal

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

    def is_directed(self, recompute=False):
        if hasattr(self, '_directed') and not recompute:
            return self._directed

        self._directed = (self.W != self.W.T).nnz != 0
        return self._directed

    def _get_upper_bound(self):
        if self.lap_type == 'normalized' or self.lap_type == 'adjacency':
            return 2
        elif self.lap_type == 'combinatorial':
            bounds = []
            # Equal for full graphs.
            bounds += [self.n_vertices * np.max(self.W)]
            # Gershgorin circle theorem. Equal for regular bipartite graphs.
            # Special case of the below bound.
            bounds += [2 * np.max(self.dw)]
            # Anderson, Morley, Eigenvalues of the Laplacian of a graph.
            # Equal for regular bipartite graphs.
            if self.n_edges > 0:
                sources, targets, _ = self.get_edge_list()
                bounds += [np.max(self.dw[sources] + self.dw[targets])]
            # Merris, A note on Laplacian graph eigenvalues.
            if not self.is_directed():
                W = self.W
            else:
                W = gsp.utils.symmetrize(self.W, method='average')
            if not np.allclose(self.dw, 0):
                m = W.dot(self.dw) / self.dw  # Mean degree of adjacent vertices.
                bounds += [np.max(self.dw + m)]
            # Good review: On upper bounds for Laplacian graph eigenvalues.
            return min(bounds)
        else:
            return np.inf

    def has_loops(self):
        return np.any(self.W.diagonal() != 0)

    def compute_laplacian(self, lap_type='combinatorial'):
        if lap_type != self.lap_type:
            # Those attributes are invalidated when the Laplacian is changed.
            # Alternative: don't allow the user to change the Laplacian.
            self._lmin = None
            self._lmax = None
            if hasattr(self, '_U'):
                del self._U
            if hasattr(self, '_e'):
                del self._e
            if hasattr(self, '_mu'):
                del self._mu
            if hasattr(self, '_D'):
                del self._D

        self.lap_type = lap_type

        if lap_type == 'combinatorial':
            D = sparse.diags(self.dwq)
            if self.s == 1:
                self.L = D - self.Wq
            elif self.s == -1:
                self.L = D + self.Wq
            elif self.s == 0:
                self.L = sparse.identity(self.N)
            else:
                I = sparse.identity(self.N)
                self.L = I - self.s * self.Wq + self.s * self.s * (D - I)
        elif lap_type == 'normalized':
            D_neg_sqrt, disconnected = util.power_diagonal(self.dwq, -0.5)
            if self.s == 1:
                self.L = sparse.identity(self.N) - D_neg_sqrt * self.Wq * D_neg_sqrt
            elif self.s == -1:
                self.L = sparse.identity(self.N) + D_neg_sqrt * self.Wq * D_neg_sqrt
            else:
                I = sparse.identity(self.N)
                D = sparse.diags(self.dwq)
                self.L = D_neg_sqrt @ (I - self.s * self.Wq + self.s * self.s * (D - I)) @ D_neg_sqrt
            self.L[disconnected, disconnected] = 0
            self.L.eliminate_zeros()
        elif lap_type == 'combinatorial directed':
            if self.is_directed():
                self.phi = util.perron_vector(self.W)
            else:
                self.phi = self.dw/self.dw.sum()
            self.P = sparse.csr_matrix(self.W/self.W.sum(axis=1), copy=False)
            Phi = sparse.diags(self.phi)
            Phi_P = Phi @ self.P
            self.L = Phi - (Phi_P + Phi_P.T.conj())/2
        elif lap_type == 'normalized directed':
            if self.is_directed():
                self.phi = util.perron_vector(self.W)
            else:
                self.phi = self.dw/self.dw.sum()
            self.P = sparse.csr_matrix(self.W/self.W.sum(axis=1), copy=False)
            Phi_sqrt, disconnected = util.power_diagonal(self.phi, 0.5)
            Phi_neg_sqrt, _ = util.power_diagonal(self.phi, -0.5)
            Phi_P_Phi = Phi_sqrt @ self.P @ Phi_neg_sqrt
            self.L = sparse.identity(self.N) - (Phi_P_Phi + Phi_P_Phi.T.conj())/2
            self.L[disconnected, disconnected] = 0
            self.L.eliminate_zeros()
        elif lap_type == 'adjacency':
            if not self.is_connected():
                raise ValueError('Laplacian does not support direct sum decomposition - calculate for each connected component explicitly')

            self.Wq_norm = util.operator_norm(self.Wq)
            self.L = sparse.identity(self.N) - self.Wq/self.Wq_norm
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
        return BigGraph(adjacency, lap_type=self.lap_type, s=self.s, q=self.q, coords=coords, plotting=self.plotting)

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
            self._Wq, self._dwq = util.magneticAdjacencyMatrix(self, self.q)
        return self._Wq

    @property
    def d(self):
        if self._d is None:
            if not self.is_directed():
                # Shortcut for undirected graphs.
                self._d = self.W.getnnz(axis=1)
                # axis=1 faster for CSR (https://stackoverflow.com/a/16391764)
            else:
                degree_in = self.W.getnnz(axis=0)
                degree_out = self.W.getnnz(axis=1)
                self._d = (degree_in + degree_out) / 2
        return self._d

    @property
    def dw(self):
        if self._dw is None:
            if not self.is_directed():
                # Shortcut for undirected graphs.
                self._dw = np.ravel(self.W.sum(axis=0))
            else:
                degree_in = np.ravel(self.W.sum(axis=0))
                degree_out = np.ravel(self.W.sum(axis=1))
                self._dw = (degree_in + degree_out) / 2
        return self._dw

    @property
    def dwq(self):
        if self._dwq is None:
            self._Wq, self._dwq = util.magneticAdjacencyMatrix(self, self.q)
        return self._dwq

    @property
    def lmin(self):
        """
        Smallest non-zero eigenvalue.
        np.nan if all the eigenvalues are zero.
        """
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
            self._lmin = util.estimate_lmin(self)

    @property
    def lmax(self):
        if self._lmax is None:
            self.logger.warning('The largest eigenvalue G.lmax is not '
                                'available, we need to estimate it. '
                                'Explicitly call G.estimate_lmax() or '
                                'G.compute_fourier_basis() '
                                'once beforehand to suppress the warning.')
            self.estimate_lmax()
        return self._lmax

    def estimate_lmax(self, method='lanczos'):
        if self._lmax_method != method:
            self._lmax = util.estimate_lmax(self, method=method)
            self._lmax_method = method

    def count_components(self):
        if self.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        if self._n_connected is None:
            self._n_connected = util.count_components(self)
            if not hasattr(self, '_connected'):
                self._connected = (self._n_connected == 1)

        return self._n_connected

    def extract_components(self):
        if self.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        subgraphs = util.extract_components(self)

        if self._n_connected is None:
            self._n_connected = len(subgraphs)
            if not hasattr(self, '_connected'):
                self._connected = (self._n_connected == 1)

        return subgraphs

    def compute_differential_operator(self):
        sources, targets, weights = self.get_edge_list()

        n = self.n_edges
        columns = np.concatenate([sources, targets])
        rows = np.concatenate([np.arange(n), np.arange(n)])
        values = np.empty(2*n)

        if self.lap_type == 'combinatorial':
            values[:n] = np.sqrt(weights)
            values[n:] = -values[:n]
        elif self.lap_type == 'normalized':
            values[:n] = +np.sqrt(weights / self.dw[sources])
            values[n:] = -np.sqrt(weights / self.dw[targets])
        else:
            raise ValueError('Unknown lap_type {}'.format(self.lap_type))

        if self.is_directed():
            values /= np.sqrt(2)

        self._D = sparse.csc_matrix((values, (rows, columns)),
                                    shape=(self.n_edges, self.n_vertices))
        self._D.eliminate_zeros()  # Self-loops introduce stored zeros.

    def _check_signal(self, s):
        r"""Check if signal is valid."""
        s = np.asanyarray(s)
        if s.shape[0] != self.n_vertices:
            raise ValueError('First dimension must be the number of vertices '
                             'G.N = {}, got {}.'.format(self.N, s.shape))
        return s

    def dirichlet_energy(self, x):
        x = self._check_signal(x)
        return x.T.dot(self.L.dot(x))


class BipartiteGraph(BigGraph):
    def __init__(self, W, lap_type='combinatorial', s=1, q=0, coords=None, plotting={}):
        super().__init__(W, lap_type=lap_type, s=s, q=q, coords=coords, plotting=plotting)
        if self.lap_type == 'normalized':
            self._lmax = 2

    def compute_fourier_basis(self, recompute=False, spectrum_only=False):
        LGraphFourier.compute_fourier_basis(self, recompute=recompute, spectrum_only=spectrum_only)
        if self.lap_type == 'normalized':
            assert self.e[-1] == 2

