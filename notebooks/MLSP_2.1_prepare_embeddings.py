import sys
sys.path.append('../')
from hypgs.models.hsn_pyg import HSN
from hypgs.utils.plot import plot_wavelets, get_hyperedge_pos_df, compute_enlarged_hulls, plot_hulls
from hypgs.utils.graph_build import calcualte_voronoi_from_coords, build_graph_from_cell_coords, build_graph_from_voronoi_polygons, build_voronoi_polygon_to_cell_mapping, plot_voronoi_polygons, plot_graph, assign_attributes
from hypgs.models.hsn_pyg import HyperScatteringModule
from hypgs.utils.data import get_hyperedge_index
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx
from hypgs.utils.data import HGDataset
from dhg import Hypergraph
import scprep
import matplotlib.pyplot as plt
import re
import phate
from phate.cluster import kmeans
from scipy.spatial import ConvexHull
from torch_geometric.data.hypergraph_data import HyperGraphData
from tqdm import tqdm
from torch_geometric.nn import MessagePassing

import argparse
# add arg
parser = argparse.ArgumentParser()
parser.add_argument('--section', type=str, required=True)
section = parser.parse_args().section

class CountCellTypesConv(MessagePassing):
    """
    Count celltypes of neighbors by message passing
    """
    def __init__(self):
        # Initialize with sum aggregation
        super(CountCellTypesConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j
    
from scipy.spatial.distance import pdist, squareform, cdist
import pathlib
data_path = '../data/hgsae2/'
pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

# for section in tqdm(['H20.33.001.CX28.MTG.02.007.1.02.02', 'H20.33.044.Cx26.MTG.02.007.1.01.04', 'H21.33.023.Cx26.MTG.02.007.1.03.05', 'H21.33.019.Cx30.MTG.02.007.5.01.01', 'H21.33.022.Cx26.MTG.02.007.2.M.04', 'H21.33.028.CX28.MTG.02.007.1.01.01']):
adata = sc.read_h5ad(f"/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/sea_ad/data/hypergraphae/{section}")

coordinates = adata.obsm['spatial']
coordinates = coordinates - coordinates.min(axis=0)
xmax, ymax = coordinates.max(axis=0)
voronoi_polygons = calcualte_voronoi_from_coords(coordinates[:, 0], coordinates[:, 1])
cell_data = pd.DataFrame(np.c_[adata.obs.index, coordinates], columns=['CELL_ID', 'X', 'Y'])
G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)
G = assign_attributes(G, cell_data, node_to_cell_mapping)

X = adata.X.toarray()
data = from_networkx(G)
data.x = torch.tensor(X)
original_dataset = [data]
to_hg_func = lambda g: Hypergraph.from_graph_kHop(g, 1)
dataset = HGDataset(original_dataset, to_hg_func)

eidf = get_hyperedge_pos_df(dataset[0], coordinates)
enlarged_hulls = compute_enlarged_hulls(eidf, )
hulls = []
df = eidf
for set_index, group in df.groupby('he_idx'):
    points = group[['x', 'y']].values
    hull = ConvexHull(points)
    hulls.append(hull)
approx_radius = [np.sqrt(hull.volume/np.pi) for hull in hulls]
approx_radius = np.array(approx_radius)
n_cells = [hull.points.shape[0] for hull in hulls]
n_cells = np.array(n_cells)
# fig, axes = plt.subplots(1,2, figsize=(12,6))
# axes[0].hist(approx_radius, 100)
# axes[0].set_xlabel('Approximate Radius')
# axes[1].hist(n_cells, 100)
# axes[1].set_xlabel('Number of Cells')

model = HyperScatteringModule(in_channels=X.shape[1], 
            trainable_laziness = False,
            trainable_scales = False, 
            activation = None, # just get one layer of wavelet transform 
            fixed_weights=True, 
            normalize='right', 
            reshape=False,
        )

s_nodes, s_edges = model(dataset[0].x.float(), dataset[0].edge_index, hyperedge_attr = dataset[0].edge_attr)
shapes = s_edges.size()
he_features = s_edges.squeeze().permute(1,0,2).reshape(shapes[2], shapes[1]*shapes[3])
# torch.save(he_features, "../data/merfish_hyperedge_GS_features.pt")

enc_df = pd.get_dummies(adata.obs[['Class', 'Subclass', 'Supertype']])
encoded_data = torch.tensor(enc_df.values, dtype=torch.float)

data = HyperGraphData(edge_index=dataset[0].edge_index, x=encoded_data)
conv = CountCellTypesConv()
node_feature_counts = conv(data.x, data.edge_index)
count_df = pd.DataFrame(node_feature_counts.cpu().numpy(), columns=enc_df.columns).astype(int)
count_df.to_csv('../data/merfish_hyperedge_celltypes.csv', index=False)

class_dict = {
    'oligo': ['Oligodendrocyte', 'OPC'],
    'astro': ['Astrocyte'],
    'immune': ['Microglia-PVM'],
    'vascular': ['Endothelial', 'VLMC'],
    'neural': ['L2/3 IT','L6 IT','L4 IT','L5 IT',
        'L5 ET','L6 CT','L6b','L6 IT Car3','L5/6 NP',
        'Lamp5 Lhx6','Lamp5','Pax6','Sncg','Vip',
        'Sst Chodl','Sst','Pvalb','Chandelier']
}

count_coarse = pd.DataFrame(index=count_df.index)
for key, value in class_dict.items():
    value = ['Subclass_'+v for v in value]
    value = [v for v in value if v in count_df.columns]
    count_coarse[key] = count_df[value].sum(axis=1)
rate_coarse = count_coarse/n_cells.reshape(-1,1)
# binarize the types into 2^5=32 types
count_coarse_bin = (count_coarse > 0).astype(int)
count_coarse_bin_combine = count_coarse_bin.copy()
count_coarse_bin_combine['class'] = count_coarse_bin_combine.applymap(str).sum(axis=1)
count_coarse_bin_combine['class'].value_counts()

heclass = count_coarse_bin_combine['class'].values

phate_op = phate.PHATE(n_components=3)
he_phate = phate_op.fit_transform(he_features.detach().numpy())

xor_dist = 1 - (heclass.reshape(1,-1) == heclass.reshape (-1,1))
epsilon = 1e-7 # like phate.
log001p_dist = squareform(pdist(np.log(rate_coarse + epsilon)))
pca_loadings = phate_op.graph.data_nu
train_mask = np.zeros(len(heclass)).astype(bool)
np.random.seed(42)
train_idx = np.random.choice(len(heclass), int(len(heclass) * 0.8), replace=False)
train_mask[train_idx] = True

data_dict = dict(
    data=pca_loadings,
    phate=he_phate,
    dist=log001p_dist,
    colors=heclass,
    xor_dist=xor_dist,
    is_train=train_mask,
)
np.savez(f"{data_path}/{section}.npz", **data_dict)
