import numpy as np
from scipy import sparse


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
    W = G.Wq if hasattr(G, 'Wq') else magneticAdjacencyMatrix(G, q=0)[0]
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
    W = G.Wq if hasattr(G, 'Wq') else magneticAdjacencyMatrix(G, q=0)[0]
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
    if N > 100:
        approx_evecs = np.empty((N, 2))
        # 0-eigenvector (exact)
        approx_evecs[:,0] = 1
        # 1st non-zero eigenvector (guess)
        idxs = np.arange(N)
        one_idxs = np.random.choice(idxs, N//2, replace=False)
        neg_one_idxs = idxs[np.isin(idxs, one_idxs, assume_unique=True, invert=True)]
        approx_evecs[one_idxs,1] = 1
        approx_evecs[neg_one_idxs,1] = -1
        if N&1:
            approx_evecs[neg_one_idxs[-1],1] = 0
    
        # simple pre-conditioner
        M = sparse.spdiags(1/G.L.diagonal(), 0, N, N)
        evals, _ = sparse.linalg.lobpcg(G.L, approx_evecs, M=M, largest=False, maxiter=maxiter)
        lmin = evals[1]
    elif N > 1:
        # if small then do exact calculation
        G.compute_fourier_basis(spectrum_only=True)
        lmin = G.lmin
    else:
        # single vertex has eigenvalue zero
        return np.nan

    assert lmin >= 0, "Smallest eigenvalue is negative {}".format(lmin)
    if np.isclose(lmin, 0):
        raise ValueError("Second eigenvalue is (close to) zero: {}".format(lmin))
    return lmin


def estimate_lmin(G, maxiter=2000):
    if G.L.shape == (1,1):
        return _estimate_lmin(G, maxiter)

    lmins = []
    for subG in G.extract_components():
        if subG.N > 1:  # skip trivial components
            lmin = _estimate_lmin(subG, maxiter)
            if not np.isnan(lmin):
                lmins.append(lmin)
    return sorted(lmins)[0] if lmins else np.nan


def estimate_lmax(G, method='lanczos'):
    if G.L.shape == (1,1):
        return G.L[0,0]

    if method == 'lanczos':
        try:
            lmax = sparse.linalg.eigsh(G.L, k=1, tol=5e-3,
                                       ncv=min(G.N, 10),
                                       return_eigenvectors=False)
            lmax = lmax[0]
            lmax *= 1.01  # Increase by 1 percent to be robust to errors.
            upper_bound = G._get_upper_bound()
            if lmax > upper_bound:
                lmax = upper_bound
        except sparse.linalg.ArpackNoConvergence:
            G.logger.warning('Lanczos method did not converge. '
                                'Using an alternative method.')
            lmax = estimate_lmax(G, method='bounds')
    elif method == 'bounds':
        lmax = G._get_upper_bound()
    else:
        raise ValueError('Unknown method {}'.format(method))

    return lmax


def power_diagonal(a, exp):
    a_exp = np.zeros(a.shape[0])
    disconnected = np.isclose(a, 0)
    np.power(a, exp, where=~disconnected, out=a_exp)
    return sparse.diags(a_exp), disconnected


def perron_vector(P):
    _evals, evecs = sparse.linalg.eigs(P.T, k=1, v0=np.ones(P.shape[0]))
    p = evecs[:,0]
    assert np.allclose(p.imag, 0)
    return p.real/p.real.sum()


def operator_norm(W, maxiter=1000):
    if W.shape == (1,1):
        return W[0,0]
    else:
        svals = sparse.linalg.svds(W.asfptype(), k=1, return_singular_vectors=False, solver="lobpcg", maxiter=maxiter)
        return svals[0]


def count_negatives(W):
    if W.has_canonical_format:
        return np.count_nonzero(W.data < 0)
    else:
        return (W < 0).nnz
