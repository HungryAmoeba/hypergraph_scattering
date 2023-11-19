"""
Helper for using dataset and dataloader in pytorch geometric.
see https://github.com/pyg-team/pytorch_geometric/blob/cf24b4bcb4e825537ba08d8fc5f31073e2cd84c7/torch_geometric/data/hypergraph_data.py
"""
import torch
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch_geometric.data import Dataset
from dhg import Graph, Hypergraph

def get_hyperedge_index(HG):
    """
    Get the hyperedge index from a hypergraph object. for the HyperGraphData class.
    
    Args:
        HG: Hypergraph object
    """
    hyperedge_list = HG.e[0]
    # Flatten the list of tuples and also create a corresponding index list
    flattened_list = []
    index_list = []
    for i, t in enumerate(hyperedge_list):
        flattened_list.extend(t)
        index_list.extend([i] * len(t))

    # Convert to 2D numpy array
    hyperedge_index = torch.tensor([flattened_list, index_list])

    return hyperedge_index

def get_HyperGraphData(HG, node_features, hyperedge_attr, labels):
    """
    Get the HyperGraphData class from a hypergraph object and the corresponding node features, hyperedge attributes and labels.
    
    Args:
        HG: Hypergraph object
        node_features (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        hyperedge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`.
            (default: :obj:`None`)
        labels (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
    
    Returns:
        a HyperGraphData object
    """
    hyperedge_index = get_hyperedge_index(HG)
    return HyperGraphData(x=node_features, edge_index=hyperedge_index, edge_attr=hyperedge_attr, y=labels)

def get_HG_data_list(original_dataset, to_hg_func=lambda g: Hypergraph.from_graph_kHop(g, k=1)):
    hgdataset = []
    for graph_dat in original_dataset:
        # Access the first graph
        # Get the edge list
        edge_list = graph_dat.edge_index.t()
        num_vertices = graph_dat.num_nodes
        # Get node features
        node_features = graph_dat.x
        # Get labels
        labels = graph_dat.y
        G = Graph(num_vertices, edge_list)
        HG1 = to_hg_func(G)
        X, lbl = node_features, labels
        Y = torch.zeros(HG1.num_e, X.shape[1]) # use all zero hyperedge attributes
        hgdataset.append(get_HyperGraphData(HG1, X, Y, lbl))
    return hgdataset

class HGDataset(Dataset):
    def __init__(self, original_dataset, to_hg_func=lambda g: Hypergraph.from_graph_kHop(g, k=1), transform=None, pre_transform=None):
        super(HGDataset, self).__init__('.', transform, pre_transform)
        self.original_dataset = original_dataset
        self.to_hg_func = to_hg_func
        self.data_list = get_HG_data_list(original_dataset, to_hg_func)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
