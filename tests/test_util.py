import unittest

from scipy import sparse
from sgw_tools import util


class TestCase(unittest.TestCase):
    def test_count_negatives(self):
        adjacency = [
            [0., 3., 0., -2.],
            [3., 0., 4., 0.],
            [0., 4., 0., 5.],
            [2., 0., -5., 0.],
        ]
        W = sparse.csr_matrix(adjacency)
        self.assertEqual(util.count_negatives(W), 2)

    def test_operator_norm(self):
        W = sparse.csr_matrix([[4.1]])
        self.assertEqual(util.operator_norm(W), 4.1)
