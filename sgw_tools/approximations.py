import numpy as np
from scipy import optimize
from scipy import sparse
from pygsp.utils import filterbank_handler
from . import util


@filterbank_handler
def compute_cheby_coeff(f, m=30, N=None, domain=None, *args, **kwargs):
    r"""
    Compute Chebyshev coefficients for a Filterbank.

    Parameters
    ----------
    f : Filter
        Filterbank with at least 1 filter
    m : int
        Maximum order of Chebyshev coeff to compute
        (default = 30)
    N : int
        Grid order used to compute quadrature
        (default = m + 1)
    i : int
        Index of the Filterbank element to compute
        (default = 0)

    Returns
    -------
    c : ndarray
        Matrix of Chebyshev coefficients

    """
    G = f.G
    i = kwargs.pop('i', 0)

    if not N:
        N = m + 1

    a_arange = domain if domain else [0, G.lmax]

    a1 = (a_arange[1] - a_arange[0]) / 2
    a2 = (a_arange[1] + a_arange[0]) / 2
    c = np.zeros(m + 1)

    tmpN = np.arange(N)
    num = np.cos(np.pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2. / N * np.dot(f._kernels[i](a1 * num + a2),
                               np.cos(np.pi * o * (tmpN + 0.5) / N))

    return c


def cheby_op(G, c, signal, domain=None, **kwargs):
    r"""
    Chebyshev polynomial of graph Laplacian applied to vector.

    Parameters
    ----------
    G : Graph
    c : ndarray or list of ndarrays
        Chebyshev coefficients for a Filter or a Filterbank
    signal : ndarray
        Signal to filter

    Returns
    -------
    r : ndarray
        Result of the filtering

    """
    # Handle if we do not have a list of filters but only a simple filter in cheby_coeff.
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    c = np.atleast_2d(c)
    Nscales, M = c.shape

    if M < 2:
        raise TypeError("The coefficients have an invalid shape")

    # thanks to that, we can also have 1d signal.
    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((G.N * Nscales, Nv))
    except IndexError:
        r = np.zeros((G.N * Nscales))

    a_arange = domain if domain else [0, G.lmax]

    a1 = float(a_arange[1] - a_arange[0]) / 2.
    a2 = float(a_arange[1] + a_arange[0]) / 2.

    twf_old = signal
    twf_cur = (G.L.dot(signal) - a2 * signal) / a1

    tmpN = np.arange(G.N, dtype=int)
    for i in range(Nscales):
        r[tmpN + G.N*i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

    factor = 2/a1 * (G.L - a2 * sparse.eye(G.N))
    for k in range(2, M):
        twf_new = factor.dot(twf_cur) - twf_old
        for i in range(Nscales):
            r[tmpN + G.N*i] += c[i, k] * twf_new

        twf_old = twf_cur
        twf_cur = twf_new

    return r


def compute_cayley_coeff(f, m, method="complex"):
    G = f.G
    if method == "complex":
        res = optimize.minimize(cayley_loss, x0=np.array([1]), args=(f,m), bounds=((0, None),))
        h = res.x
        z = util.cayley_transform(h*G.e)
        y = f.evaluate(G.e).squeeze()
        p = np.polynomial.polynomial.Polynomial.fit(z, y, m, domain=[0, G.lmax], window=[0, G.lmax])
        return h[0], np.real(p.coef[0]), p.coef[1:]/2
    elif method == "real":
        y = f.evaluate(G.e).squeeze()
        initial_guess = [1] + [1] + [0]*(m-1)
        fit = optimize.curve_fit(util.cayley_filter, G.e, y, p0=initial_guess, jac=util.cayley_filter_jac)
        coeffs = fit[0]
        return coeffs[0], coeffs[1], coeffs[2:]
    else:
        raise ValueError("Unsupported method")


def cayley_loss(h, f, m):
    G = f.G
    z = util.cayley_transform(h*G.e)
    y = f.evaluate(G.e).squeeze()
    p = np.polynomial.polynomial.Polynomial.fit(z, y, m, domain=[0, G.lmax], window=[0, G.lmax])
    p.coef[0] = np.real(p.coef[0])
    return np.linalg.norm(y - p(z))
