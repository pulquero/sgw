import unittest

import numpy as np
import scipy.linalg
from scipy import sparse
import networkx as nx

from pygsp import graphs
from sgw_tools import BigGraph
from sgw_tools import RingGraph, StarGraph, DirectedPath, RandomRegular


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._G = BigGraph.create_from(graphs.Logo())
        cls._G.compute_fourier_basis()
        cls._G.compute_differential_operator()

        cls._rng = np.random.default_rng(42)
        cls._signal = cls._rng.uniform(size=cls._G.N)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_graph(self):
        adjacency = [
            [0., 3., 0., 2.],
            [3., 0., 4., 0.],
            [0., 4., 0., 5.],
            [2., 0., 5., 0.],
        ]

        # Input types.
        G = BigGraph(adjacency)
        self.assertIs(type(G.W), sparse.csr_matrix)
        adjacency = np.array(adjacency)
        G = BigGraph(adjacency)
        self.assertIs(type(G.W), sparse.csr_matrix)
        adjacency = sparse.coo_matrix(adjacency)
        G = BigGraph(adjacency)
        self.assertIs(type(G.W), sparse.csr_matrix)
        adjacency = sparse.csr_matrix(adjacency)

        # Attributes.
        np.testing.assert_allclose(G.W.toarray(), adjacency.toarray())
        np.testing.assert_allclose(G.A.toarray(), G.W.toarray() > 0)
        np.testing.assert_allclose(G.d, np.array([2, 2, 2, 2]))
        np.testing.assert_allclose(G.dw, np.array([5, 7, 9, 7]))
        self.assertEqual(G.n_vertices, 4)
        self.assertIs(G.N, G.n_vertices)
        self.assertEqual(G.n_edges, 4)
        self.assertIs(G.Ne, G.n_edges)

        # Errors and warnings.
        self.assertRaises(ValueError, BigGraph, np.ones((3, 4)))
        self.assertRaises(ValueError, BigGraph, np.ones((3, 3, 4)))
        self.assertRaises(ValueError, BigGraph, [[0, np.nan], [0, 0]])
        self.assertRaises(ValueError, BigGraph, [[0, np.inf], [0, 0]])
        with self.assertLogs(level='WARNING'):
            BigGraph([[0, -1], [-1, 0]])
        with self.assertLogs(level='WARNING'):
            BigGraph([[1, 1], [1, 0]])
        for attr in ['A', 'd', 'dw', 'lmax', 'U', 'e', 'coherence', 'D']:
            self.assertRaises(AttributeError, setattr, G, attr, None)
            self.assertRaises(AttributeError, delattr, G, attr)

    def test_degree(self):
        graph = BigGraph([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        np.testing.assert_allclose(graph.d, [1, 2, 1])
        np.testing.assert_allclose(graph.dw, [1, 3, 2])
        graph = BigGraph([
            [0, 1, 0],
            [0, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        np.testing.assert_allclose(graph.d, [0.5, 1.5, 1])
        np.testing.assert_allclose(graph.dw, [0.5, 2.5, 2])

    def test_is_connected(self):
        graph = BigGraph([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        self.assertEqual(graph.is_connected(), True)
        graph = BigGraph([
            [0, 1, 0],
            [1, 0, 0],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        self.assertEqual(graph.is_connected(), False)
        graph = BigGraph([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        self.assertEqual(graph.is_connected(), False)
        graph = BigGraph([
            [0, 1, 0],
            [0, 0, 2],
            [3, 0, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        self.assertEqual(graph.is_connected(), True)

    def test_is_directed(self):
        graph = BigGraph([
            [0, 3, 0, 0],
            [3, 0, 4, 0],
            [0, 4, 0, 2],
            [0, 0, 2, 0],
        ])
        assert graph.W.nnz == 6
        self.assertEqual(graph.is_directed(), False)

    def test_laplacian(self):
        G = BigGraph([
            [0, 3, 0, 1],
            [3, 0, 1, 0],
            [0, 1, 0, 3],
            [1, 0, 3, 0],
        ])
        laplacian = np.array([
            [+4, -3, +0, -1],
            [-3, +4, -1, +0],
            [+0, -1, +4, -3],
            [-1, +0, -3, +4],
        ])
        self.assertFalse(G.is_directed())
        G.compute_laplacian('combinatorial')
        np.testing.assert_allclose(G.L.toarray(), laplacian)
        G.compute_laplacian('normalized')
        np.testing.assert_allclose(G.L.toarray(), laplacian/4)
    
        G = BigGraph([
            [0, 6, 0, 1],
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [1, 0, 3, 0],
        ])
        self.assertTrue(G.is_directed())
        G.compute_laplacian('combinatorial')
        np.testing.assert_allclose(G.L.toarray(), laplacian)
        G.compute_laplacian('normalized')
        np.testing.assert_allclose(G.L.toarray(), laplacian/4)
    
        def test_combinatorial(G):
            np.testing.assert_equal(G.L.toarray(), G.L.T.toarray())
            np.testing.assert_equal(G.L.sum(axis=0), 0)
            np.testing.assert_equal(G.L.sum(axis=1), 0)
            np.testing.assert_equal(G.L.diagonal(), G.dw)
    
        def test_normalized(G):
            np.testing.assert_equal(G.L.toarray(), G.L.T.toarray())
            np.testing.assert_equal(G.L.diagonal(), 1)
    
        def test_adjacency(G):
            np.testing.assert_equal(G.L.diagonal(), 1)
    
        G = BigGraph.create_from(graphs.ErdosRenyi(100, directed=False))
        self.assertFalse(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        test_combinatorial(G)
        G.compute_laplacian(lap_type='normalized')
        test_normalized(G)
        G.compute_laplacian(lap_type='adjacency')
        test_adjacency(G)
    
        G = BigGraph.create_from(graphs.ErdosRenyi(100, directed=True))
        self.assertTrue(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        test_combinatorial(G)
        G.compute_laplacian(lap_type='normalized')
        test_normalized(G)
        G.compute_laplacian(lap_type='adjacency')
        test_adjacency(G)

    def test_estimate_lmin(self):
        graph = BigGraph([
            [0, 3, 0, 1],
            [3, 0, 1, 0],
            [0, 1, 0, 3],
            [1, 0, 3, 0],
        ])
        graph.estimate_lmin()
        np.testing.assert_approx_equal(graph.lmin, 2)

        graph = BigGraph.create_from(graphs.Sensor(N=107, seed=27))
        np.testing.assert_approx_equal(graph.lmin, 0.08818931)

        # isolated node
        for lap_type in ['combinatorial', 'normalized', 'adjacency']:
            graph = BigGraph([[3]], lap_type=lap_type)
            np.testing.assert_approx_equal(graph.lmin, np.nan)

    def test_estimate_lmax(self):
        graph = BigGraph.create_from(graphs.Sensor())
        self.assertRaises(ValueError, graph.estimate_lmax, method='unk')
    
        def check_lmax(graph, lmax):
            graph.estimate_lmax(method='bounds')
            np.testing.assert_allclose(graph.lmax, lmax, err_msg='bounds', atol=1e-15)
            graph.estimate_lmax(method='lanczos')
            np.testing.assert_allclose(graph.lmax, lmax, err_msg='lanczos', atol=1e-15)
            graph.compute_fourier_basis()
            np.testing.assert_allclose(graph.lmax, lmax, err_msg='fourier', atol=1e-15)
    
        # Full graph (bound is tight).
        n_nodes, value = 10, 2
        adjacency = np.full((n_nodes, n_nodes), value)
        graph = BigGraph(adjacency, lap_type='combinatorial')
        check_lmax(graph, lmax=value*n_nodes)
    
        # Regular bipartite graph (bound is tight).
        adjacency = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
        graph = BigGraph(adjacency, lap_type='combinatorial')
        check_lmax(graph, lmax=4)
    
        # Bipartite graph (bound is tight).
        adjacency = [
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        graph = BigGraph(adjacency, lap_type='normalized')
        check_lmax(graph, lmax=2)

        # isolated node
        for lap_type in ['combinatorial', 'normalized', 'adjacency']:
            graph = BigGraph([[3]], lap_type=lap_type)
            check_lmax(graph, lmax=0)

    def test_fourier_basis(self):
        # Smallest eigenvalue close to zero.
        np.testing.assert_allclose(self._G.e[0], 0, atol=1e-12)
        # First eigenvector is constant.
        N = self._G.N
        # note: eigenvector sign might be flipped
        np.testing.assert_allclose(np.abs(self._G.U[:, 0]), np.sqrt(N) / N)
        # Spectrum bounded by [0, 2] for the normalized Laplacian.
        G = BigGraph.create_from(graphs.Logo(), lap_type='normalized')
        # check full eigendecomposition
        G.compute_fourier_basis()
        assert len(G.e) == G.N
        assert G.U.shape[1] == G.N
        assert G.e[-1] < 2
    
    def test_eigendecompositions(self):
        G = BigGraph.create_from(graphs.Logo())
        U1, e1, V1 = scipy.linalg.svd(G.L.toarray())
        U2, e2, V2 = np.linalg.svd(G.L.toarray())
        e3, U3 = np.linalg.eig(G.L.toarray())
        e4, U4 = scipy.linalg.eig(G.L.toarray())
        e5, U5 = np.linalg.eigh(G.L.toarray())
        e6, U6 = scipy.linalg.eigh(G.L.toarray())
    
        def correct_sign(U):
            signs = np.sign(U[0, :])
            signs[signs == 0] = 1
            return U * signs
        U1 = correct_sign(U1)
        U2 = correct_sign(U2)
        U3 = correct_sign(U3)
        U4 = correct_sign(U4)
        U5 = correct_sign(U5)
        U6 = correct_sign(U6)
        V1 = correct_sign(V1.T)
        V2 = correct_sign(V2.T)
    
        inds3 = np.argsort(e3)[::-1]
        inds4 = np.argsort(e4)[::-1]
        np.testing.assert_allclose(e2, e1)
        np.testing.assert_allclose(e3[inds3], e1, atol=1e-12)
        np.testing.assert_allclose(e4[inds4], e1, atol=1e-12)
        np.testing.assert_allclose(e5[::-1], e1, atol=1e-12)
        np.testing.assert_allclose(e6[::-1], e1, atol=1e-12)
        np.testing.assert_allclose(U2, U1, atol=1e-12)
        np.testing.assert_allclose(V1, U1, atol=1e-12)
        np.testing.assert_allclose(V2, U1, atol=1e-12)
        np.testing.assert_allclose(U3[:, inds3], U1, atol=1e-10)
        np.testing.assert_allclose(U4[:, inds4], U1, atol=1e-10)
        np.testing.assert_allclose(U5[:, ::-1], U1, atol=1e-10)
        np.testing.assert_allclose(U6[:, ::-1], U1, atol=1e-10)
    
    def test_fourier_transform(self):
        s = self._rng.uniform(size=(self._G.N, 99, 21))
        s_hat = self._G.gft(s)
        s_star = self._G.igft(s_hat)
        np.testing.assert_allclose(s, s_star, rtol=1e-6)

    def test_edge_list(self):
        for directed in [False, True]:
            G = BigGraph.create_from(graphs.ErdosRenyi(100, directed=directed))
            sources, targets, weights = G.get_edge_list()
            if not directed:
                self.assertTrue(np.all(sources <= targets))
            edges = np.arange(G.n_edges)
            np.testing.assert_equal(G.W[sources[edges], targets[edges]],
                                    weights[edges][np.newaxis, :])

    def test_differential_operator(self, n_vertices=98):
        r"""The Laplacian must always be the divergence of the gradient,
        whether the Laplacian is combinatorial or normalized, and whether the
        graph is directed or weighted."""
        def test_incidence_nx(graph):
            r"""Test that the incidence matrix corresponds to NetworkX."""
            incidence_pg = -np.sign(graph.D.T.toarray())
            G = nx.DiGraph if graph.is_directed() else nx.Graph
            graph_nx = nx.from_scipy_sparse_array(graph.W, create_using=G)
            incidence_nx = nx.incidence_matrix(graph_nx, oriented=True)
            np.testing.assert_equal(incidence_pg, incidence_nx.toarray())
        for graph in [
                      BigGraph(np.zeros((n_vertices, n_vertices))),
                      BigGraph(np.identity(n_vertices)),
                      BigGraph([[0, 0.8], [0.8, 0]]),
                      BigGraph([[1.3, 0], [0.4, 0.5]]),
                      BigGraph.create_from(graphs.ErdosRenyi(n_vertices, directed=False, seed=42)),
                      BigGraph.create_from(graphs.ErdosRenyi(n_vertices, directed=True, seed=42))
                      ]:
            for lap_type in ['combinatorial', 'normalized']:
                graph.compute_laplacian(lap_type)
                graph.compute_differential_operator()
                L = graph.D.T.dot(graph.D)
                np.testing.assert_allclose(L.toarray(), graph.L.toarray())
                test_incidence_nx(graph)

    def test_difference(self):
        for lap_type in ['combinatorial', 'normalized']:
            G = BigGraph.create_from(graphs.Logo(), lap_type=lap_type)
            G.compute_differential_operator()
            y = G.grad(self._signal)
            self.assertEqual(len(y), G.n_edges)
            z = G.div(y)
            self.assertEqual(len(z), G.n_vertices)
            np.testing.assert_allclose(z, G.L.dot(self._signal))

    def test_dirichlet_energy(self, n_vertices=100):
        r"""The Dirichlet energy is defined as the norm of the gradient."""
        signal = np.random.default_rng(42).uniform(size=n_vertices)
        for lap_type in ['combinatorial', 'normalized']:
            graph = BigGraph.create_from(graphs.BarabasiAlbert(n_vertices), lap_type=lap_type)
            graph.compute_differential_operator()
            energy = graph.dirichlet_energy(signal)
            grad_norm = np.sum(graph.grad(signal)**2)
            np.testing.assert_allclose(energy, grad_norm)

    def test_empty_graph(self, n_vertices=11):
        """Empty graphs have either no edge, or self-loops only. The Laplacian
        doesn't see self-loops, as the gradient on those edges is always zero.
        """
        adjacencies = [
            np.zeros((n_vertices, n_vertices)),
            np.identity(n_vertices),
        ]
        for adjacency, n_edges in zip(adjacencies, [0, n_vertices]):
            graph = BigGraph(adjacency)
            self.assertEqual(graph.n_vertices, n_vertices)
            self.assertEqual(graph.n_edges, n_edges)
            self.assertEqual(graph.W.nnz, n_edges)
            for laplacian in ['combinatorial', 'normalized']:
                graph.compute_laplacian(laplacian)
                self.assertEqual(graph.L.nnz, 0)
                sources, targets, weights = graph.get_edge_list()
                self.assertEqual(len(sources), n_edges)
                self.assertEqual(len(targets), n_edges)
                self.assertEqual(len(weights), n_edges)
                graph.compute_differential_operator()
                self.assertEqual(graph.D.nnz, 0)
                graph.compute_fourier_basis()
                np.testing.assert_allclose(graph.U, np.identity(n_vertices))
                np.testing.assert_allclose(graph.e, np.zeros(n_vertices))
            # NetworkX uses the same conventions.
            G = nx.from_scipy_sparse_array(graph.W)
            self.assertEqual(nx.laplacian_matrix(G).nnz, 0)
            self.assertEqual(nx.normalized_laplacian_matrix(G).nnz, 0)
            self.assertEqual(nx.incidence_matrix(G).nnz, 0)

    def test_adjacency_types(self, n_vertices=10):
    
        rng = np.random.default_rng(42)
        W = 10 * np.abs(rng.normal(size=(n_vertices, n_vertices)))
        W = W + W.T
        W = W - np.diag(np.diag(W))
    
        def test(adjacency):
            G = BigGraph(adjacency)
            G.compute_laplacian('combinatorial')
            G.compute_laplacian('normalized')
            G.estimate_lmax()
            G.compute_fourier_basis()
            G.compute_differential_operator()
    
        test(W)
        #test(W.astype(np.float32))
        test(W.astype(int))
        test(sparse.csr_matrix(W))
        #test(sparse.csr_matrix(W, dtype=np.float32))
        test(sparse.csr_matrix(W, dtype=int))
        test(sparse.csc_matrix(W))
        test(sparse.coo_matrix(W))

    def test_ring_even_combinatorial(self):
        self._test_ring(20, 'combinatorial')

    def test_ring_odd_combinatorial(self):
        self._test_ring(19, 'combinatorial')

    def test_ring_even_normalized(self):
        self._test_ring(20, 'normalized')

    def test_ring_odd_normalized(self):
        self._test_ring(19, 'normalized')

    def _test_ring(self, n=20, lap_type='combinatorial'):
        graph = RingGraph(n, lap_type=lap_type)
        self.assertEqual(graph.n_vertices, n)
        self.assertEqual(graph.n_edges, n)
        np.testing.assert_array_equal(graph.d, n * [2])
        np.testing.assert_allclose(np.linalg.norm(graph.coords[1:], axis=1), 1)
        graph.compute_fourier_basis()
        expected_evals = scipy.linalg.eigh(graph.L.toarray(), eigvals_only=True)
        expected_evals[0] = 0
        np.testing.assert_allclose(graph.e, expected_evals)
        # check eigenvectors
        L = graph.L
        U = graph.U
        for i, e in enumerate(graph.e):
            np.testing.assert_almost_equal(U.T@U, np.eye(n), 10)
            np.testing.assert_almost_equal(L @ U[:,i], e * U[:,i], 10)

    def test_star(self, n=20):
        graph = StarGraph(n)
        self.assertEqual(graph.n_vertices, n)
        self.assertEqual(graph.n_edges, n-1)
        np.testing.assert_array_equal(graph.d, [n-1] + (n-1) * [1])
        np.testing.assert_allclose(np.linalg.norm(graph.coords[1:], axis=1), 1)

    def test_randomregular(self):
        k = 6
        G = RandomRegular(k=k)
        np.testing.assert_equal(G.W.sum(0), k)
        np.testing.assert_equal(G.W.sum(1), k)

    def test_directed_path(self, n=5):
        graph = DirectedPath(n)
        graph.set_coordinates('line2D')
        adjacency = [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        coords = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
        np.testing.assert_array_equal(graph.W.toarray(), adjacency)
        np.testing.assert_array_equal(graph.coords, coords)


if __name__ == '__main__':
    unittest.main()
