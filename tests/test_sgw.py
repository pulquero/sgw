import unittest

import numpy as np
import scipy
import sgw_tools as sgw
import pygsp as gsp


class TestCase(unittest.TestCase):
    def test_tig(self):
        G = sgw.BigGraph([[0,1],[1,0]])
        g = gsp.filters.Heat(G)
        s = sgw.createSignal(G)
        sgw._tig(g, s)

        # single signal
        s = np.array([[1],[-1]])
        sgw._tig(g, s)

        # multiple signals
        s = np.array([[1,2,3],[-1,-2,-3]])
        sgw._tig(g, s)

        # multiple filters, single signal
        g = gsp.filters.Heat(G, tau=[1,5])
        s = np.array([[1],[-1]])
        sgw._tig(g, s)

        # isolated node
        G = sgw.BigGraph([[2]])
        g = gsp.filters.Heat(G)
        s = sgw.createSignal(G)
        sgw._tig(g, s)

        # single signal
        s = np.array([1])
        sgw._tig(g, s)

        # multiple signals
        s = np.array([[1,2,3]])
        sgw._tig(g, s)

        # multiple filters, single signal
        g = gsp.filters.Heat(G, tau=[1,5])
        s = np.array([1])
        sgw._tig(g, s)

        # multiple filters, multiple signals
        g = gsp.filters.Heat(G, tau=[1,5])
        s = np.array([[1,2,3]])
        sgw._tig(g, s)

    def test_chebyshev_filter(self):
        G = gsp.graphs.Sensor(100, lap_type="normalized", seed=5)
        func = lambda x: np.exp(-x**2)
        g = sgw.CustomFilter(G, func)

        signal = np.ones(G.N)
        order = 20
        expected = g.filter(signal, order=order)

        domain = [0, 2]
        func_e = func(G.e)

        gsp_coeffs = sgw.approximations.compute_cheby_coeff(g, m=order, domain=domain)
        gsp_g = sgw.ChebyshevFilter(G, gsp_coeffs, domain, "pygsp")
        np.testing.assert_allclose(gsp_g.evaluate(G.e).squeeze(), func_e, err_msg="pygsp evaluate")
        gsp_actual = gsp_g.filter(signal)
        np.testing.assert_allclose(gsp_actual, expected, err_msg="pygsp coeffs")

        np_cheby = np.polynomial.Chebyshev.fit(G.e, func(G.e), deg=order, domain=domain)
        np_g = sgw.ChebyshevFilter(G, np_cheby.coef, domain, "numpy")
        np.testing.assert_allclose(np_g.evaluate(G.e).squeeze(), func_e, err_msg="numpy evaluate")
        np_actual = np_g.filter(signal)
        np.testing.assert_allclose(np_actual, expected, err_msg="numpy coeffs")

    def test_cayley_filter(self):
        G = gsp.graphs.Sensor(100, lap_type="normalized", seed=5)
        G.compute_fourier_basis()
        func = lambda x: np.exp(-x**2)
        g = sgw.CustomFilter(G, func)
        signal = np.ones(G.N)
        expected = g.filter(signal, method="exact")

        order = 20
        h, c0, c = sgw.approximations.compute_cayley_coeff(g, order, method="real")

        g = sgw.CayleyFilter(G, np.array([h, c0] + list(c)))
        np.testing.assert_allclose(g.evaluate(G.e).squeeze(), func(G.e))
        actual = g.filter(signal, method="exact")
        np.testing.assert_allclose(actual, expected)
        actual = g.filter(signal, method="cayley-exact")
        np.testing.assert_allclose(actual, expected)
        actual = g.filter(signal, method="cayley")
        np.testing.assert_allclose(actual, expected)

    def test_cayley_multifilter(self):
        G = gsp.graphs.Sensor(100, lap_type="normalized", seed=5)
        G.compute_fourier_basis()
        coeffs = [
            [1, 1, 0, 0, 1],
            [1, 0, 1, -1],
        ]
        g = sgw.CayleyFilter(G, coeffs)
        signal = np.ones(G.N)
        expected = g.filter(signal, method="exact")
        out1 = g.filter(signal, method="cayley-exact")
        out2 = g.filter(signal, method="cayley")
        np.testing.assert_allclose(out1, expected, err_msg="cayley-exact")
        np.testing.assert_allclose(out2, expected, err_msg="cayley", atol=2e-5)

    def test_cayley_filter_multisignal(self):
        G = gsp.graphs.Sensor(100, lap_type="normalized", seed=5)
        G.compute_fourier_basis()
        coeffs = np.array([2, 0, 1, 0, -1])
        g = sgw.CayleyFilter(G, coeffs)
        signals = np.array([np.ones(G.N), np.random.default_rng(seed=89).random(G.N)]).T
        expected = g.filter(signals, method="exact")
        out1 = g.filter(signals, method="cayley-exact")
        out2 = g.filter(signals, method="cayley")
        np.testing.assert_allclose(out1, expected, err_msg="cayley-exact")
        np.testing.assert_allclose(out2, expected, err_msg="cayley", atol=2e-5)

    def test_cayley_multifilter_multisignal(self):
        G = gsp.graphs.Sensor(100, lap_type="normalized", seed=5)
        G.compute_fourier_basis()
        coeffs = [
            [1, 1, 0, 0, 1],
            [1, 0, 1, -1],
            [2, 0, 0, 1],
        ]
        g = sgw.CayleyFilter(G, coeffs)
        signals = np.array([np.ones(G.N), np.random.default_rng(seed=89).random(G.N)]).T
        expected = g.filter(signals, method="exact")
        out1 = g.filter(signals, method="cayley-exact")
        out2 = g.filter(signals, method="cayley")
        np.testing.assert_allclose(out1, expected, err_msg="cayley-exact")
        np.testing.assert_allclose(out2, expected, err_msg="cayley", atol=3e-5)
