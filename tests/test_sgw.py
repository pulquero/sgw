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
