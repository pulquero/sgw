import numpy as np
import pygsp as gsp
from . import util


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

        kernels = [
            np.polynomial.Chebyshev(coeffs, domain=domain) for coeffs in kernel_coeffs
        ]
        super().__init__(G, kernels)

    def filter(self, s, method='chebyshev', order=30):
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
                s = gsp.filters.approximations.cheby_op(self.G, c, s)
                s = s.reshape((self.G.N, n_features_out, n_signals), order='F')
                s = s.swapaxes(1, 2)

            elif n_features_in == self.Nf:  # Synthesis.
                s = s.swapaxes(1, 2)
                s_in = s.reshape(
                    (self.G.N * n_features_in, n_signals), order='F')
                s = np.zeros((self.G.N, n_signals))
                tmpN = np.arange(self.G.N, dtype=int)
                for i in range(n_features_in):
                    s += gsp.filters.approximations.cheby_op(self.G,
                                                 c[i],
                                                 s_in[i * self.G.N + tmpN])
                s = np.expand_dims(s, 2)

        else:
            raise ValueError('Unknown method {}.'.format(method))

        # Return a 1D signal if e.g. a 1D signal was filtered by one filter.
        return s.squeeze()


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
