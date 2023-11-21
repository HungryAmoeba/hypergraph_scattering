from torch_geometric.utils import to_undirected
import dhg 
import torch
from torch_geometric.utils import from_networkx

import networkx as nx
import matplotlib.pyplot as plt

def data_to_hg(data, add_k_hop = 0, min_k_hop_size = 0):
    edge_index_undirected = to_undirected(data.edge_index)
    unique_edges = edge_index_undirected.t().tolist()

    edges_as_tuples = [(int(edge[0]), int(edge[1])) for edge in unique_edges]
    if add_k_hop:
        g = dhg.Graph(data.num_nodes, edges_as_tuples)
        hg_k_hop = dhg.Hypergraph.from_graph_kHop(g, k = add_k_hop)
        hyperedges = hg_k_hop.e 
        if min_k_hop_size > 0:
            hyperedges_filtered = [edge for edge in hyperedges[0] if len(edge) > min_k_hop_size]
            edges_as_tuples = edges_as_tuples + hyperedges_filtered
        else:
            edges_as_tuples = edges_as_tuples + hyperedges[0]
    hg = dhg.Hypergraph(data.num_nodes, edges_as_tuples)
    return hg

if __name__ == '__main__': 
    # visualize converting an ER graph into a hypergraph with the desired features

    # Create a random graph using NetworkX
    G = nx.fast_gnp_random_graph(10, 0.3)  # Generate a random graph with 10 nodes and edge probability 0.3

    # Visualize the generated graph (optional)
    nx.draw(G, with_labels=True)
    plt.show()

    # Convert NetworkX graph to PyTorch Geometric data object
    data = from_networkx(G)

    hg = data_to_hg(data, add_k_hop=1, min_k_hop_size = 3)
    hg.draw()