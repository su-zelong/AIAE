import torch

def create_adj(edges):
    num_nodes = edges.max().item() + 1
    row_indices = edges[0, :]
    col_indices = edges[1, :]
    values = torch.ones(edges.size(1), dtype=torch.float32)
    adjacency_matrix = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]), values, (num_nodes, num_nodes))
    dense_adjacency_matrix = adjacency_matrix.to_dense()
    return dense_adjacency_matrix