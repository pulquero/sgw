from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch_geometric.utils as torch_utils
from torch_geometric.typing import OptTensor
from sgw_tools import util

from . import layers

def get_edge_tensors(G):
    edge_list = G.get_edge_list()
    edge_index = np.array([
        np.concatenate((edge_list[0], edge_list[1])),
        np.concatenate((edge_list[1], edge_list[0]))
    ])
    edge_weight = np.concatenate((edge_list[2], edge_list[2]))
    return torch.from_numpy(edge_index).long(), torch.from_numpy(edge_weight)


def get_laplacian(edge_index: Tensor,
        edge_weight: OptTensor = None,
        lap_type: str = "normalized",
        dtype: Optional[torch.dtype] = None,
        num_nodes: Optional[int] = None):
    if lap_type == "adjacency":
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size[1], dtype=dtype, device=edge_index.device)
        op_norm = util.operator_norm(torch_utils.to_scipy_sparse_matrix(edge_index, edge_weight))
        num_nodes = torch_utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
        return torch_utils.add_self_loops(edge_index, -edge_weight/op_norm, fill_value=1.0, num_nodes=num_nodes)
    else:
        normalization_mapping = {
            "combinatorial": None,
            "normalized": "sym"
        }
        normalization = normalization_mapping[lap_type]
        return torch_utils.get_laplacian(edge_index, edge_weight=edge_weight, normalization=normalization, dtype=dtype, num_nodes=num_nodes)
    