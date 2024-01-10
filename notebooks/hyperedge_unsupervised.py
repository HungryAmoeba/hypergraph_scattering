import sys
from dhg import Hypergraph
from hypgs.models.hsn_pyg import HSN
from hypgs.utils.data import HGDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import random_split
from pytorch_lightning.loggers import TensorBoardLogger
from hypgs import DATA_DIR
import anndata
import os
import torch
import numpy as np
import phate
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
import random 
from torch_geometric.utils.convert import from_networkx
from importlib import reload
from hypgs.models.hsn_pyg import HyperScatteringModule
from einops import rearrange
from hypgs.utils.hypergraph_interpretability import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import pdist, squareform

data_path = os.path.join(DATA_DIR, 'spatial_data', 'A23-290-CP_anndata.h5ad')
cell_data = anndata.read_h5ad(data_path)
print(cell_data)

data_path_diseased = os.path.join(DATA_DIR, 'spatial_data', 'YA-7-CP_anndata.h5ad')
cell_data_diseased = anndata.read_h5ad(data_path_diseased)
print(cell_data_diseased)

shared_elements = cell_data.var_names.intersection(cell_data_diseased.var_names)
print(shared_elements)

cell_data = cell_data[:, cell_data.var_names.isin(shared_elements)]
cell_data_diseased = cell_data_diseased[:, cell_data_diseased.var_names.isin(shared_elements)]

print(cell_data)
print(cell_data_diseased)

epithelial_cell_markers = ['MUC1', 'EPCAM', 'CDH1']
'''
mucin 1 (MUC1) aka epithelial membrane antigen (EMA)
epithelial cell adhesion molecule (Ep-CAM), also known as CD326
E-cadherin (CDH1)
____alternatives____
polymorphic epithelial mucin (PEM)

'''
pericyte_markers = ['PDGFRB', 'PECAM1', 'CSPG4']

'''
PECAM1 aka CD31 platelet and endothelial cell adhesion molecule, excludes endothelial cells
CSPG4 chondroitin sulfate proteoglycan 4 [ (human)]
'''

#____Immune_markers____


pan_Tcell_markers = ['CD3E', 'CD3D']
#All T cells express CD3, which has subunits encoded by different genes (delta, epsilon, zeta)

CD4_Tcell_markers = ['CD4']
# subsets
Teff =  ['IFNG','TNF','CXCR3','TBX21']
Treg =  ['IL10','TGFB1','IL7R','IL2RA', 'CTLA4', 'FOXP3']
Tfh = ['BCL6', 'PDCD1', 'CXCR5']

CD8_Tcell_markers = ['CD8A']
# subsets
Teff =  ['KLRG1', 'IFNG', 'TNF','CXCR3','TBX21']
Texhausted = ['PDCD1','LAG3','CTLA4','TIM3']
Tmem = ['IL7R', 'CD45RO', 'EOMES']
Trm = ['SELL', 'CD69', 'ITGAE', 'ITGA1', 'CCR7', 'SIPR1']

all_markers = epithelial_cell_markers + pericyte_markers + pan_Tcell_markers + CD4_Tcell_markers + CD8_Tcell_markers
all_subsets = Teff + Treg + Tfh + Teff + Texhausted + Tmem + Trm

all_markers_and_subsets = all_markers + all_subsets
print('There are {} markers and subsets in total'.format(len(all_markers_and_subsets)))
print(all_markers_and_subsets)

cell_data_markers = cell_data[:, cell_data.var_names.isin(all_markers_and_subsets)]
cell_data_diseased_markers = cell_data_diseased[:, cell_data_diseased.var_names.isin(all_markers_and_subsets)]

# check to see which var_names in all_markers_and_subsets are not in cell_data_markers
missing_markers = [marker for marker in all_markers_and_subsets if marker not in cell_data_markers.var_names]
print(f"there are {len(missing_markers)} missing markers in cell_data_markers")
print(missing_markers)

print(cell_data_markers)
print(cell_data_diseased_markers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scales = [0,1,2,4]
model = HyperScatteringModule(in_channels=1, # doesn't matter here.
            trainable_laziness = False,
            trainable_scales = False, 
            activation = None, # just get one layer of wavelet transform 
            fixed_weights=True, 
            normalize='right', 
            reshape=False,
            scale_list = scales,
    ).to(device)

# do everything on a subset of the data called tiny
num_points = 1500
cell_data_tiny = cell_data[:num_points, :]
G = get_graph(cell_data_tiny, K = 5)
edge_feat, dataset = get_hyperedge_reps(model, cell_data_tiny, G, features='full_seq_phate')
print(edge_feat.shape)
edge_feat = edge_feat.reshape(edge_feat.shape[2], -1)
phate_result, clusters = plot_hypergraph_features_3d(edge_feat, K = 5)
hyperedge_stats, hyperedge_spatial = get_hyperedge_statistics(dataset)

clf = train_decision_tree(hyperedge_stats[0][0].numpy(), clusters)
tree.plot_tree(clf, filled=True)

# create a new figure and visualize the spatial coordinates and color by cluster
# visualize the spatial coordinates and color by cluster
spatial = hyperedge_spatial[0].detach().numpy()

plt.figure()
plt.scatter(spatial[:, 0], spatial[:, 1], c=clusters)
plt.colorbar()
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Spatial Coordinates Colored by Cluster')
plt.show()
plt.savefig('spatial_clusters.png')

# construct a distance matrix from spatial
# Assuming `spatial` is a numpy array containing the spatial coordinates
distance_matrix = torch.tensor(squareform(pdist(spatial)), dtype = torch.float32)
# scale the distance matrix to have min 0 and max 1
distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())


import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, int_dim = 25):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int_dim),
            nn.ReLU(),
            nn.Linear(int_dim, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, int_dim),
            nn.ReLU(),
            nn.Linear(int_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class EdgeFeatDataset(Dataset):
    def __init__(self, edge_feat):
        self.edge_feat = edge_feat
    
    def __len__(self):
        return len(self.edge_feat)
    
    def __getitem__(self, idx):
        return self.edge_feat[idx]
    
# Create dataset
dataset = EdgeFeatDataset(edge_feat)

# Set hyperparameters
hidden_dim = 2
input_dim = edge_feat.shape[-1]

# Create Autoencoder instance
AE = Autoencoder(input_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(), lr=0.001)

# Train Autoencoder
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# don't do any batching, and just go for it ig
def distance_loss(encoded, distance_matrix):
    pairwise_distances = torch.cdist(encoded, encoded)
    loss = criterion(pairwise_distances, distance_matrix)
    return loss

alpha = 2
edge_feat = torch.tensor(edge_feat, dtype = torch.float32)

# first train the model on distance loss only
for epoch in range(4000):
    total_loss = 0

    optimizer.zero_grad()
    inputs = edge_feat
    encoded, outputs = AE(inputs)
    loss_distance = distance_loss(encoded, distance_matrix)
    loss = loss_distance
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    # print the distance loss for the epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}: Distance Loss = {loss_distance.item():.4f}")

print('starting training on both reconstruction and distance loss')
for epoch in range(300):
    total_loss = 0

    optimizer.zero_grad()
    inputs = edge_feat
    encoded, outputs = AE(inputs)
    loss_reconstruction = criterion(outputs, inputs)
    #import pdb; pdb.set_trace()
    loss_distance = distance_loss(encoded, distance_matrix)
    loss = loss_reconstruction + alpha * loss_distance
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    # print the distance and reconstruction loss for the epoch
    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}: Reconstruction Loss = {loss_reconstruction.item():.4f}, Distance Loss = {loss_distance.item():.4f}")


# for epoch in range(10):
#     total_loss = 0
#     import pdb; pdb.set_trace()
#     for data in dataset_w_ind:
#         print(data.shape)
#         import pdb; pdb.set_trace()
#         # optimizer.zero_grad()
#         # inputs = data.float()
#         # outputs = AE(inputs)
#         # loss = criterion(outputs, inputs)
#         # loss.backward()
#         # optimizer.step()
#         # total_loss += loss.item()
#     print(f"Epoch {epoch+1}: Training Loss = {total_loss/len(dataloader):.4f}")

print("Autoencoder training completed, with distance preservation.")

# Get latent space representation
latent_space = AE.encoder(torch.Tensor(edge_feat)).detach().numpy()
print(latent_space.shape)
#visualize latent space in 3D
import matplotlib.pyplot as plt
# Get latent space representation
# make 2d scatter plot of latent space
plt.figure()
plt.scatter(latent_space[:, 0], latent_space[:, 1], c = clusters)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('AE Embedding, colored by PHATE cluster')
plt.savefig('latent_space2d.png')

# # Create a 3D scatter plot of the latent space
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2])
# ax.set_xlabel('Latent Dimension 1')
# ax.set_ylabel('Latent Dimension 2')
# ax.set_zlabel('Latent Dimension 3')
# ax.set_title('Latent Space Scatter Plot')
# plt.savefig('latent_space.png')

import pdb; pdb.set_trace()

# # visualize the spatial coordinates and color by cluster
# spatial = hyperedge_spatial[0].detach().numpy()
# #import pdb; pdb.set_trace()
# # plot the 2D spatial coordinates and color by cluster
# plt.scatter(np.linspace(0, 10, 5), np.ones(5))

# import pdb; pdb.set_trace()
# plt.scatter(spatial[:,0], spatial[:,1])
# #plt.savefig('spatial_clusters.png')


print('finished running hypergraph interpretability on tiny dataset')

# idea: contrastive loss to preserve distance in embedding and reconstruction


