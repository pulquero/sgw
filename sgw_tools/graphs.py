import numpy as np
import pygsp as gsp
from scipy import sparse
from .graph import BigGraph

class StarGraph(gsp.graphs.Comet):
    def __init__(self, N, **kwargs):
        super().__init__(N, N-1, **kwargs)
        self.n_vertices = self.N
        self.n_edges = self.Ne


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


class RandomRegular(BigGraph):
    def __init__(self, N=64, k=6, max_iter=10, seed=None, **kwargs):

        self.k = k
        self.max_iter = max_iter
        self.seed = seed

        rng = np.random.default_rng(seed)

        # continue until a proper graph is formed
        if (N * k) % 2 == 1:
            raise ValueError("input error: N*d must be even!")

        # a list of open half-edges
        U = np.kron(np.ones(k, dtype=np.int32), np.arange(N, dtype=np.int32))

        # the graphs adjacency matrix
        A = sparse.lil_matrix(np.zeros((N, N)))

        edgesTested = 0
        repetition = 1

        while np.size(U) and repetition < max_iter:
            edgesTested += 1

            if edgesTested % 5000 == 0:
                self.logger.debug("createRandRegGraph() progress: edges= "
                                  "{}/{}.".format(edgesTested, N*k/2))

            # chose at random 2 half edges
            i1 = rng.integers(0, U.shape[0])
            i2 = rng.integers(0, U.shape[0])
            v1 = U[i1]
            v2 = U[i2]

            # check that there are no loops nor parallel edges
            if v1 == v2 or A[v1, v2] == 1:
                # restart process if needed
                if edgesTested == N*k:
                    repetition = repetition + 1
                    edgesTested = 0
                    U = np.kron(np.ones(k, dtype=np.int32), np.arange(N, dtype=np.int32))
                    A = sparse.lil_matrix(np.zeros((N, N)))
            else:
                # add edge to graph
                A[v1, v2] = 1
                A[v2, v1] = 1

                # remove used half-edges
                v = sorted([i1, i2])
                U = np.concatenate((U[:v[0]], U[v[0] + 1:v[1]], U[v[1] + 1:]))

        super(RandomRegular, self).__init__(A, **kwargs)

        self.is_regular()

    def is_regular(self):
        r"""
        Troubleshoot a given regular graph.
        """
        warn = False
        msg = 'The given matrix'

        # check symmetry
        if np.abs(self.A - self.A.T).sum() > 0:
            warn = True
            msg = '{} is not symmetric,'.format(msg)

        # check parallel edged
        if self.A.max(axis=None) > 1:
            warn = True
            msg = '{} has parallel edges,'.format(msg)

        # check that d is d-regular
        if np.min(self.d) != np.max(self.d):
            warn = True
            msg = '{} is not d-regular,'.format(msg)

        # check that g doesn't contain any self-loop
        if self.A.diagonal().any():
            warn = True
            msg = '{} has self loop.'.format(msg)

        if warn:
            self.logger.warning('{}.'.format(msg[:-1]))

    def _get_extra_repr(self):
        return dict(k=self.k, seed=self.seed)
