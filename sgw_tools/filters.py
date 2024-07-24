import numpy as np
from scipy import sparse
import pygsp as gsp
from . import util
from . import approximations


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
                lmin = util.estimate_lmin(G, maxiter=maxiter)
            else:
                nze = G.e[np.invert(np.isclose(G.e, 0))]
                lmin = nze[0] if len(nze) > 0 else np.nan
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


class ChebyshevFilter(gsp.filters.Filter):
    def __init__(self, G, coeff_bank, domain, coeff_normalization="pygsp"):
        coeff_bank = np.asanyarray(coeff_bank)
        if coeff_bank.ndim == 1:
            coeff_bank = coeff_bank.reshape(1, -1)

        if coeff_normalization == "numpy":
            self.coeff_bank = np.array(coeff_bank, copy=True)
            self.coeff_bank[:, 0] *= 2
            kernel_coeffs = coeff_bank
        elif coeff_normalization == "pygsp":
            self.coeff_bank = coeff_bank
            kernel_coeffs = np.array(coeff_bank, copy=True)
            kernel_coeffs[:, 0] /= 2
        else:
            raise ValueError(f"Invalid coefficient normalization: {coeff_normalization}")

        self.domain = domain

        kernels = [
            np.polynomial.Chebyshev(coeffs, domain=domain) for coeffs in kernel_coeffs
        ]
        super().__init__(G, kernels)

    def filter(self, s, method='chebyshev'):
        if s.shape[0] != self.G.N:
            raise ValueError('First dimension should be the number of nodes '
                             'G.N = {}, got {}.'.format(self.G.N, s.shape))

        # TODO: not in self.Nin (Nf = Nin x Nout).
        if s.ndim == 1 or s.shape[-1] not in [1, self.Nf]:
            if s.ndim == 3:
                raise ValueError('Third dimension (#features) should be '
                                 'either 1 or the number of filters Nf = {}, '
                                 'got {}.'.format(self.Nf, s.shape))
            s = np.expand_dims(s, -1)
        n_features_in = s.shape[-1]

        if s.ndim < 3:
            s = np.expand_dims(s, 1)
        n_signals = s.shape[1]

        if s.ndim > 3:
            raise ValueError('At most 3 dimensions: '
                             '#nodes x #signals x #features.')
        assert s.ndim == 3

        # TODO: generalize to 2D (m --> n) filter banks.
        # Only 1 --> Nf (analysis) and Nf --> 1 (synthesis) for now.
        n_features_out = self.Nf if n_features_in == 1 else 1

        if method == 'exact':

            # TODO: will be handled by g.adjoint().
            axis = 1 if n_features_in == 1 else 2
            f = self.evaluate(self.G.e)
            f = np.expand_dims(f.T, axis)
            assert f.shape == (self.G.N, n_features_in, n_features_out)

            s = self.G.gft(s)
            s = np.matmul(s, f)
            s = self.G.igft(s)

        elif method == 'chebyshev':

            c = self.coeff_bank

            if n_features_in == 1:  # Analysis.
                s = s.squeeze(axis=2)
                s = approximations.cheby_op(self.G, c, s, domain=self.domain)
                s = s.reshape((self.G.N, n_features_out, n_signals), order='F')
                s = s.swapaxes(1, 2)

            elif n_features_in == self.Nf:  # Synthesis.
                s = s.swapaxes(1, 2)
                s_in = s.reshape(
                    (self.G.N * n_features_in, n_signals), order='F')
                s = np.zeros((self.G.N, n_signals))
                tmpN = np.arange(self.G.N, dtype=int)
                for i in range(n_features_in):
                    s += approximations.cheby_op(self.G,
                                                 c[i],
                                                 s_in[i * self.G.N + tmpN],
                                                 domain=self.domain)
                s = np.expand_dims(s, 2)

        else:
            raise ValueError('Unknown method {}.'.format(method))

        # Return a 1D signal if e.g. a 1D signal was filtered by one filter.
        return s.squeeze()


class CayleyFilter(gsp.filters.Filter):
    def __init__(self, G, coeff_bank):
        if type(coeff_bank) == np.ndarray:
            if coeff_bank.ndim == 1:
                coeff_bank = coeff_bank.reshape(1, -1)
            else:
                coeff_bank = [np.polynomial.polyutils.trimcoef(coeffs) for coeffs in coeff_bank]
        self.coeff_bank = coeff_bank

        kernels = []
        for coeffs in self.coeff_bank:
            assert np.isreal(coeffs[0]), "h must be real"
            assert np.isreal(coeffs[1]), "0th coefficient must be real"
            kernel = lambda x, h=np.real(coeffs[0]), c0=np.real(coeffs[1]), c=coeffs[2:]: util.cayley_filter(x, h, c0, *c)
            kernels.append(kernel)
        super().__init__(G, kernels)

    def filter(self, s, method='cayley', maxiter=1000):
        if s.shape[0] != self.G.N:
            raise ValueError('First dimension should be the number of nodes '
                             'G.N = {}, got {}.'.format(self.G.N, s.shape))

        # TODO: not in self.Nin (Nf = Nin x Nout).
        if s.ndim == 1 or s.shape[-1] not in [1, self.Nf]:
            if s.ndim == 3:
                raise ValueError('Third dimension (#features) should be '
                                 'either 1 or the number of filters Nf = {}, '
                                 'got {}.'.format(self.Nf, s.shape))
            s = np.expand_dims(s, -1)
        n_features_in = s.shape[-1]

        if s.ndim < 3:
            s = np.expand_dims(s, 1)
        n_signals = s.shape[1]

        if s.ndim > 3:
            raise ValueError('At most 3 dimensions: '
                             '#nodes x #signals x #features.')
        assert s.ndim == 3

        # TODO: generalize to 2D (m --> n) filter banks.
        # Only 1 --> Nf (analysis) and Nf --> 1 (synthesis) for now.
        n_features_out = self.Nf if n_features_in == 1 else 1

        if method == 'exact':

            # TODO: will be handled by g.adjoint().
            axis = 1 if n_features_in == 1 else 2
            f = self.evaluate(self.G.e)
            f = np.expand_dims(f.T, axis)
            assert f.shape == (self.G.N, n_features_in, n_features_out)

            s = self.G.gft(s)
            s = np.matmul(s, f)
            s = self.G.igft(s)

        elif method == 'cayley-exact':
            if n_features_in != 1:
                raise ValueError("Currently only analysis is supported")

            s = s.squeeze(axis=2)
            L = self.G.L
            im_eye = sparse.identity(self.G.N, dtype=complex) * 1j
            ys = []
            for coeffs in self.coeff_bank:
                h, c0, c = np.real(coeffs[0]), np.real(coeffs[1]), coeffs[2:]
                hL = h * L
                J = sparse.linalg.inv((hL + im_eye).tocsc()) @ (hL - im_eye)
                y = c0 * s
                v = s
                for r in range(len(c)):
                    v = J @ v
                    y += 2 * np.real(c[r] * v)
                ys.append(y)
            s = np.array(ys)
            s = s.transpose((1, 2, 0))

        elif method == 'cayley':
            if n_features_in != 1:
                raise ValueError("Currently only analysis is supported")

            s = s.squeeze(axis=2)
            L = self.G.L
            im_diag = np.ones(self.G.N)
            ys = []
            for idx, coeffs in enumerate(self.coeff_bank):
                h, c0, c = np.real(coeffs[0]), np.real(coeffs[1]), coeffs[2:]
                y = c0 * s

                # Jacobi iteration matrix
                M_diag_re = h * L.diagonal()
                M_offdiag_re = h * L - sparse.diags(M_diag_re)
                M_diag_im = im_diag

                D_denom = M_diag_re**2 + M_diag_im**2
                D_re = M_diag_re/D_denom
                D_im = -M_diag_im/D_denom
                J_re = -diagmul(D_re, M_offdiag_re)
                J_im = -diagmul(D_im, M_offdiag_re)
                B_re = sparse.diags(D_re * M_diag_re + D_im * M_diag_im) - J_re
                B_im = -sparse.diags(D_re * M_diag_im - D_im * M_diag_re) - J_im

                v_re = s
                v_im = np.zeros_like(s)
                for r in range(len(c)):
                    # Jacobi iteration
                    b_re, b_im = cmul(B_re, B_im, v_re, v_im)
                    v_new_re, v_new_im = b_re, b_im
                    iter = 0
                    while True:
                        v_re, v_im = v_new_re, v_new_im
                        v_new_re, v_new_im = cmul(J_re, J_im, v_re, v_im)
                        v_new_re, v_new_im = v_new_re + b_re, v_new_im + b_im
                        if np.allclose(v_re, v_new_re) and np.allclose(v_im, v_new_im):
                            break
                        elif iter >= maxiter:
                            raise Exception("Maximum iterations exceeded for term {} of filter {}".format(r+1, idx+1))
                        iter += 1
                    v_re, v_im = v_new_re, v_new_im
            
                    y += 2 * (np.real(c[r]) * v_re - np.imag(c[r]) * v_im)
                ys.append(y)
            s = np.array(ys)
            s = s.transpose((1, 2, 0))

        else:
            raise ValueError('Unknown method {}.'.format(method))

        # Return a 1D signal if e.g. a 1D signal was filtered by one filter.
        return s.squeeze()


def cmul(ar, ai, br, bi):
    return (ar@br - ai@bi, ar@bi + ai@br)


def diagmul(d, M):
    assert sparse.isspmatrix_csr(M)
    new_data = np.empty_like(M.data)
    for i in range(M.shape[0]):
        idxs = slice(M.indptr[i], M.indptr[i+1])
        new_data[idxs] = d[i] * M.data[idxs]
    return sparse.csr_matrix((new_data, M.indices, M.indptr), shape=M.shape)


class CustomFilter(gsp.filters.Filter):
    def __init__(self, G, funcs, scales=1):
        if not hasattr(funcs, '__iter__'):
            funcs = [funcs]

        if not hasattr(scales, '__iter__'):
            scales = [scales]
        self.scales = scales

        kernels = []
        for s in scales:
            for func in funcs:
                kernels.append(lambda x, s=s, func=func: func(x*s))
        super().__init__(G, kernels)
