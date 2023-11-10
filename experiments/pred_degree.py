from hypergraph_scattering.models.hyper_scattering_net import HSN
import dhg 
import torch 
from typing import Tuple, Optional 

# how to create a 

#dhg.random.hypergraph_Gnm(num_v, num_e, method='low_order_first', prob_k_list=None)

num_v = 10
num_e = 12

hg = dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first')

dhg.visualization.draw_hypergraph(hg)

import pdb; pdb.set_trace()

