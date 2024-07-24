from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, ones
from torch_geometric.typing import OptTensor
import torch_geometric.utils as tg_utils
import sgw_torch
import numpy as np


class SpectralLayer(MessagePassing):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        lap_type: str,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lap_type = lap_type

    def get_laplacian(self,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        lap_type: str = "normalized",
        dtype: Optional[torch.dtype] = None,
        num_nodes: Optional[int] = None
    ):
        return sgw_torch.get_laplacian(edge_index, edge_weight=edge_weight, lap_type=lap_type, dtype=dtype, num_nodes=num_nodes)

    def get_lambda_max(self, edge_weight: Tensor, num_nodes: int) -> Tensor:
        if self.lap_type == "normalized" or self.lap_type == "adjacency":
            return torch.tensor(2.0, dtype=edge_weight.dtype)
        elif self.lap_type == "combinatorial":
            D = edge_weight[edge_weight > 0]
            A = -edge_weight[edge_weight < 0]
            return min(num_nodes*A.max(), 2.0 * D.max())
        else:
            raise ValueError(f"Unsupported lap_type {self.lap_type}")

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class ChebLayer(SpectralLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        lap_type: str ='normalized',
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, lap_type)
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: OptTensor,
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = self.get_laplacian(edge_index, edge_weight,
                                                self.lap_type, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = self.get_lambda_max(edge_weight, num_nodes)
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def _weight_op(self, x: Tensor, weight: Tensor):
        return F.linear(x, weight)

    def _evaluate_chebyshev(
        self,
        coeffs: List[Tensor],
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        out = self._weight_op(Tx_0, coeffs[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(coeffs) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self._weight_op(Tx_1, coeffs[1])

        for coeff in coeffs[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + self._weight_op(Tx_2, coeff)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:
        ws = [lin.weight for lin in self.lins]
        return self._evaluate_chebyshev(ws, x, edge_index, edge_weight, batch, lambda_max)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'lap_type={self.lap_type})')


class ChebIILayer(ChebLayer):
    """
    https://github.com/ivam-he/ChebNetII
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        lap_type: str ='normalized',
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, K, lap_type, bias)

    def reset_parameters(self):
        super().reset_parameters()
        # gamma_j should be an estimation of filter value h(x_j), which should be positive so initialize to positive value
        for lin in self.lins:
            ones(lin.weight)

    def convert_coefficients(self, ys=None):
        if ys is None:
            ys = list(self.parameters())

        k1 = len(self.lins)

        def transform(Ts):
            itr = zip(ys, Ts)
            first = next(itr)
            w = first[0] * first[1]  # initialise with correct shape
            for y, T in itr:
                w += y*T
            return 2*w/k1

        ws = []
        x = torch.from_numpy(np.polynomial.chebyshev.chebpts1(k1))
        a = torch.ones(k1)
        b = x
        ws.append(transform(a))
        ws.append(transform(b))
        for _ in range(2, k1):
            T = 2*x*b - a
            ws.append(transform(T))
            a = b
            b = T
        return ws

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:
        # gamma_j (ys[j]) should be an estimation of filter value h(x_j), which should be positive so relu it
        ys = [F.relu(lin.weight) for lin in self.lins]
        ws = self.convert_coefficients(ys)
        return self._evaluate_chebyshev(ws, x, edge_index, edge_weight, batch, lambda_max)


class CayleyLayer(SpectralLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        lap_type: str ='normalized',
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, lap_type)
        self.h = Parameter(torch.tensor(1.0))
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(2*K - 1)
        ])

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        ones(self.h)
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def _weight_op(self, x: Tensor, weight: Tensor):
        return F.linear(x, weight)

    def _evaluate_cayley(
        self,
        h: Tensor,
        coeffs: List[Tensor],
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        maxiter = 1000
    ) -> Tensor:

        num_nodes = x.size(self.node_dim)

        edge_index, edge_weight = self.get_laplacian(edge_index, edge_weight,
                                                self.lap_type, x.dtype,
                                                num_nodes)

        diag_mask = (edge_index[0] == edge_index[1])
        diag_index = edge_index[:, diag_mask]
        offdiag_mask = torch.logical_not(diag_mask)
        offdiag_index = edge_index[:, offdiag_mask]

        M_diag_re = h.item() * edge_weight[diag_mask]
        M_offdiag_re = h.item() * edge_weight[offdiag_mask]
        M_diag_im = torch.ones(num_nodes)

        D_denom = M_diag_re**2 + M_diag_im**2
        D_re = M_diag_re/D_denom
        D_im = -M_diag_im/D_denom
        J_re = -diagmul(D_re, offdiag_index, M_offdiag_re)
        J_im = -diagmul(D_im, offdiag_index, M_offdiag_re)
        B_diag_re = D_re * M_diag_re + D_im * M_diag_im
        B_diag_im = D_re * M_diag_im - D_im * M_diag_re
        B_index = torch.cat([offdiag_index, diag_index], dim=1)
        B_re = torch.cat([-J_re, B_diag_re])
        B_im = torch.cat([-J_im, -B_diag_im])

        out = self._weight_op(x, coeffs[0])

        v_re = x
        v_im = torch.zeros_like(x)
        for r in range(1, len(coeffs), 2):
            # Jacobi iteration
            b_re, b_im = self.cpropagate(B_index, x=(v_re, v_im), norm=(B_re, B_im))
            v_new_re, v_new_im = b_re, b_im
            iter = 0
            while True:
                v_re, v_im = v_new_re, v_new_im
                v_new_re, v_new_im = self.cpropagate(offdiag_index, x=(v_re, v_im), norm=(J_re, J_im))
                v_new_re, v_new_im = v_new_re + b_re, v_new_im + b_im
                if torch.allclose(v_re, v_new_re) and torch.allclose(v_im, v_new_im):
                    break
                elif iter >= maxiter:
                    raise Exception(f"Maximum iterations exceeded for term {r}")
                iter += 1
            v_re, v_im = v_new_re, v_new_im

            out += 2 * (self._weight_op(v_re, coeffs[r]) - self._weight_op(v_im, coeffs[r+1]))

        if self.bias is not None:
            out = out + self.bias

        return out

    def cpropagate(self, edge_index, x, norm):
        return (self.propagate(edge_index, x=x[0], norm=norm[0]) - self.propagate(edge_index, x=x[1], norm=norm[1]),
                self.propagate(edge_index, x=x[1], norm=norm[0]) + self.propagate(edge_index, x=x[0], norm=norm[1])
                )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        ws = [lin.weight for lin in self.lins]
        return self._evaluate_cayley(self.h, ws, x, edge_index, edge_weight)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'lap_type={self.lap_type})')


def diagmul(d, edge_index, edge_weight):
    new_weight = torch.zeros_like(edge_weight)
    for i, r in enumerate(edge_index[0]):
        new_weight[i] += d[r] * edge_weight[i]
    return new_weight
