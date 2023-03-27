import numpy as np
import pygsp as gsp


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
