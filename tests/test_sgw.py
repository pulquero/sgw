import unittest

import numpy as np
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
        signal = np.ones(G.N)
        g = sgw.CustomFilter(G, func)
        order = 20
        expected = g.filter(signal, order=order)
        func_e = func(G.e)

        gsp_coeffs = gsp.filters.compute_cheby_coeff(g, m=order)
        gsp_g = sgw.ChebyshevFilter(G, gsp_coeffs, [0, G.lmax], "pygsp")
        np.testing.assert_allclose(gsp_g.evaluate(G.e).squeeze(), func_e, err_msg="pygsp evaluate")
        gsp_actual = gsp_g.filter(signal, order=order)
        np.testing.assert_allclose(gsp_actual, expected, err_msg="pygsp coeffs")

        domain = [0, 2]
        np_cheby = np.polynomial.Chebyshev.fit(G.e, func(G.e), deg=order, domain=domain)
        np_g = sgw.ChebyshevFilter(G, np_cheby.coef, domain, "numpy")
        np.testing.assert_allclose(np_g.evaluate(G.e).squeeze(), func_e, err_msg="numpy evaluate")
        np_actual = np_g.filter(signal, order=order)
        np.testing.assert_allclose(np_actual, expected, err_msg="numpy coeffs", rtol=0.1)

