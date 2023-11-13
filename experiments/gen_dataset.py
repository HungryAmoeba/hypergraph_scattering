import os 
import sys
# fix this later!
sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
from hypg_scattering.models.hyper_scattering_net import HSN

import dhg 
import torch 
import torch.optim as optim

hg = dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first').to(device)
