import sys
import torch
import logging

from torch_geometric.datasets import Planetoid
from arguments import Argument

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def construct_edges(edges, idx):
    edges_list = edges.tolist()
    if type(idx) != list:
        idx_list = idx.tolist()
        mapping = {x: i for i, x in enumerate(idx_list)}
    else:
        idx_list = idx
        mapping = {x: x for x in idx_list}
    new_edge_start, new_edge_end = [], []
    for i in range(len(edges_list[0])):
        edge_start, edge_end = edges_list[0][i], edges_list[1][i]
        if edge_start in idx_list and edge_end in idx_list:
            new_edge_start.append(mapping[edge_start])
            new_edge_end.append(mapping[edge_end])
    new_edge = torch.tensor([new_edge_start, new_edge_end])
    return new_edge

def load_data(args):
    if not args.dataset:
        args.dataset = 'CiteSeer'
    if args.dataset not in ['CiteSeer', 'Cora', 'PubMed']:
        logger.info(f"{args.dataset} not support yet! Please select in ['CiteSeer', 'Cora', 'PubMed'] !")
        sys.exit()
    logger.info(f'Dataset has select {args.dataset}')
    logger.debug(f'Loading Dataset from ./dataset')
    dataset = Planetoid(args.CiteSeer_file, name=args.dataset)
    if not args.reconstruct_dataset:
        train_x = dataset.x[dataset.train_mask]    # num_nodes * num_node_feature
        train_y = dataset.y[dataset.train_mask]    # node label
        val_x = dataset.x[dataset.val_mask]
        val_y = dataset.y[dataset.val_mask]
        test_x = dataset.x[dataset.test_mask]
        test_y = dataset.y[dataset.test_mask]

        edges = dataset.edge_index    # [2, num_edges]
        logger.info(f'Dataset length = {len(dataset.x)}, Edge nums = {len(edges[0])}')
        logger.info(f'Train length = {len(train_x)}; Val length = {len(val_x)}; Test length = {len(test_x)}')
    else:
        # create node
        logger.debug(f'Reconstructing Trian/val/Test datset as 4:1:5 ing ... !')
        all_x = torch.cat((dataset.x[dataset.train_mask], dataset.x[dataset.val_mask], dataset.x[dataset.test_mask]), dim=0)
        all_y = torch.cat((dataset.y[dataset.train_mask], dataset.y[dataset.val_mask], dataset.y[dataset.test_mask]), dim=0)
        piece = all_y.shape[0] // 10
        train_length, val_length = piece * 4, piece
        indices = torch.randperm(all_y.size(0))
        train_x, train_y = all_x[indices[:train_length]], all_y[indices[:train_length]]
        val_x, val_y = all_x[indices[train_length:train_length+val_length]], all_y[indices[train_length:train_length+val_length]]
        test_x, test_y = all_x[indices[train_length+val_length:]], all_y[indices[train_length+val_length:]]
        assert len(test_x) + len(val_x) + len(train_x) == len(all_x), 'Split should equal !'

        # create edge
        all_edges = dataset.edge_index
        all_idx = torch.arange(len(dataset.x))
        in_idx = torch.cat((all_idx[dataset.train_mask], all_idx[dataset.val_mask], all_idx[dataset.test_mask]))
        edges = construct_edges(all_edges, in_idx)    # filter edges based on all_x
        train_edges = construct_edges(edges, indices[:train_length])

        # create mask
        train_mask = [False] * len(all_x)
        for i in indices[:train_length]:
            train_mask[i] = True
        val_mask = [False] * len(all_x)
        for i in indices[train_length: train_length + val_length]:
            val_mask[i] = True
        test_mask = [False] * len(all_x)
        for i in indices[train_length + val_length:]:
            test_mask[i] = True
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

    return (train_x, train_y, train_mask, val_x, val_y, val_mask, test_x, test_y, test_mask, train_edges, edges)

if __name__ == '__main__':
   args = Argument().get_args()
   ret = load_data(args)
