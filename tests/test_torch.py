import unittest

import numpy as np
import sgw_tools as sgw
import sgw_torch
import pygsp as gsp
import torch
import torch_geometric.utils as torch_utils


class TestCase(unittest.TestCase):
    def test_get_laplacian(self):
        G = sgw.BigGraph.create_from(gsp.graphs.Sensor(50, seed=32), lap_type="adjacency")
        edge_index, edge_weight = sgw_torch.get_edge_tensors(G)
        L_index, L_weight = sgw_torch.get_laplacian(edge_index, edge_weight, lap_type="adjacency", num_nodes=G.N)
        torch_L = torch_utils.to_scipy_sparse_matrix(L_index, L_weight)
        np.testing.assert_allclose(G.L.toarray(), torch_L.toarray())

    def test_ChebLayer(self):
        G = gsp.graphs.Sensor(34, seed=506, lap_type="normalized")
        K = 5
        domain = [0, 2]
        coeffs = sgw.approximations.compute_cheby_coeff(gsp.filters.Heat(G, tau=5), m=K, domain=domain)
        coeffs[0] /= 2

        s = sgw.createSignal(G, nodes=[0])
        g = sgw.ChebyshevFilter(G, coeffs, domain, coeff_normalization="numpy")
        expected = g.filter(s)

        layer = sgw_torch.layers.ChebLayer(1, 1, K+1, lap_type="normalized", bias=False)
        for c, p in zip(coeffs, layer.parameters()):
            p.data = torch.tensor([[c]])
        edge_index, edge_weight = sgw_torch.get_edge_tensors(G)
        y = layer(torch.from_numpy(s), edge_index, edge_weight)

        np.testing.assert_allclose(expected, y.detach().numpy().squeeze())

    def test_CayleyLayer(self):
        G = gsp.graphs.Sensor(34, seed=506, lap_type="normalized")
        K = 8
        h, c0, c = sgw.approximations.compute_cayley_coeff(gsp.filters.Heat(G, tau=5), m=K)

        s = sgw.createSignal(G, nodes=[0])
        g = sgw.CayleyFilter(G, np.array([h, c0] + list(c)))
        expected = g.filter(s)

        layer = sgw_torch.layers.CayleyLayer(1, 1, K+1, lap_type="normalized", bias=False)
        params = list(layer.parameters())
        params[0].data = torch.tensor(h)
        params[1].data = torch.tensor([[c0]])
        for i in range(len(c)):
            params[2*i+2].data = torch.tensor([[c[i].real]])
            params[2*i+3].data = torch.tensor([[c[i].imag]])
        edge_index, edge_weight = sgw_torch.get_edge_tensors(G)
        y = layer(torch.from_numpy(s), edge_index, edge_weight)

        np.testing.assert_allclose(expected, y.detach().numpy().squeeze())
