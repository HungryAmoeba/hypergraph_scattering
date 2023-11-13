#from hypg_scattering.models.hyper_scattering_net import HSN
import os 
import sys
# fix this later!
sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
from hypg_scattering.models.hyper_scattering_net import HSN

import dhg 
import torch 
import torch.optim as optim

from typing import Tuple, Optional 
from dhg.random import set_seed
from dhg import Graph
from dhg.models import GCN
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
import time
import torch.nn.functional as F
import torch.nn as nn
import random
from tqdm import trange
import matplotlib.pyplot as plt

def gen_hypg_deg_dataset(num_v, num_e, num_graphs):
    hypergraph_dataset = [dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first') for _ in range(num_graphs)]

    return hypergraph_dataset

def train_node_degree(net, hg, X, node_labels, optimizer, criterion, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    #import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
    node_pred, edge_pred = net(X, hg)

    loss = criterion(torch.squeeze(node_pred), node_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()

@torch.no_grad()
def infer(net, hg, X, node_labels, test=False):
    net.eval()
    node_pred, edge_pred = net(X,hg)
    if not test:
        #todo
        pass

if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"training on device {device}")

    num_graphs = 200 
    num_graphs_test = 100
    num_v = 10
    num_e = 20 
    # generate graph dataset
    train_dataset = gen_hypg_deg_dataset(num_v, num_e, num_graphs)
    #dataset = list(zip(hypergraphs, node_labels, edge_labels))

    test_dataset = gen_hypg_deg_dataset(num_v, num_e, num_graphs_test)
    #test_hypergraphs, test_node_labels, test_edge_labels = gen_hypg_deg_dataset(num_v, num_e, num_graphs_test)
    #test_dataset = list(zip(test_hypergraphs, test_node_labels, test_edge_labels))
    # put diracs on each node
    starting_features = torch.eye(num_v).to(device)

    signal_features = num_v
    # hidden channels argument is not applicable in this model
    # 1 for output channels (regress degree)

    hidden_channels = 16
    # all the bells and whistles
    # net = HSN(signal_features, hidden_channels, 1, activation = "modulus", trainable_scales = True, fixed_weights=False).to(device)
    # simple model
    net = HSN(signal_features, hidden_channels, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    train_losses = []
    # best performance is simple model with 2 epochs
    for epoch in trange(2):
        random.shuffle(train_dataset)
        epoch_loss = 0
        for hg in train_dataset:
            #import pdb; pdb.set_trace()
            hg = hg.to(device)
            # is the issue in this step?
            #hg = dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first').to(device)
            #hg = hg.to(device)
            
            node_label = hg.D_v.values().to(device)

            # initialize starting edge features to be a matrix of zeros
            num_e_hg = hg.num_e 
            Y = torch.zeros(num_e_hg, num_v).to(device)

            net.train()

            st = time.time()
            optimizer.zero_grad()
            node_pred, edge_pred = net(hg, starting_features, Y)
            loss = criterion(torch.squeeze(node_pred), node_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_losses.append(loss.item())

            #epoch_loss += train_node_degree(net, hg, starting_features, node_label, optimizer, criterion, epoch)
            #print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")

        #if epoch%10 == 0:
        print(f'Loss on epoch {epoch} is {epoch_loss/len(train_dataset)}')
    
    test_loss = 0
    for hg in test_dataset:
        hg = hg.to(device)
        node_label = hg.D_v.values().to(device)
        #edge_label = edge_label.to(device)

        torch.no_grad()
        net.eval()
        num_e_hg = hg.num_e 
        Y = torch.zeros(num_e_hg, num_v).to(device)
        node_pred, edge_pred = net(hg, starting_features, Y) 
        #import pdb; pdb.set_trace()
        loss = criterion(torch.squeeze(node_pred), node_label)
        test_loss += loss.item() 
    #import pdb; pdb.set_trace()


    print(net)
    print(f'nonlin is {net.activation}')
    print(f'Test loss is {test_loss/len(test_dataset)}')
    plt.plot(train_losses)
    plt.title("train losses on vertex degree pred (individual graphs)")
    plt.savefig('figures/train_losses_vertex_degree.png')

    
        


    


# # define your train function
# def train(net, X, A, lbls, train_idx, optimizer, epoch):
#     net.train()

#     st = time.time()
#     optimizer.zero_grad()
#     outs = net(X, A)
#     outs, lbls = outs[train_idx], lbls[train_idx]
#     loss = F.cross_entropy(outs, lbls)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
#     return loss.item()

# # define your validation and testing function
# @torch.no_grad()
# def infer(net, X, A, lbls, idx, test=False):
#     net.eval()
#     outs = net(X, A)
#     outs, lbls = outs[idx], lbls[idx]
#     if not test:
#         # validation with you evaluator
#         res = evaluator.validate(lbls, outs)
#     else:
#         # testing with you evaluator
#         res = evaluator.test(lbls, outs)
#     return res


# if __name__ == "__main__":
#     set_seed(2022)
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     # config your evaluation metric here
#     evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
#     # load Your data here
#     data = Cora()
#     X, lbl = data["features"], data["labels"]
#     # construct your correlation structure here
#     G = Graph(data["num_vertices"], data["edge_list"])
#     train_mask = data["train_mask"]
#     val_mask = data["val_mask"]
#     test_mask = data["test_mask"]

#     # initialize your model here
#     net = GCN(data["dim_features"], 16, data["num_classes"])
#     optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

#     X, lbl = X.to(device), lbl.to(device)
#     G = G.to(device)
#     net = net.to(device)

#     best_state = None
#     best_epoch, best_val = 0, 0
#     for epoch in range(200):
#         # train
#         train(net, X, G, lbl, train_mask, optimizer, epoch)
#         # validation
#         if epoch % 1 == 0:
#             with torch.no_grad():
#                 val_res = infer(net, X, G, lbl, val_mask)
#             if val_res > best_val:
#                 print(f"update best: {val_res:.5f}")
#                 best_epoch = epoch
#                 best_val = val_res
#                 best_state = deepcopy(net.state_dict())
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")
#     # testing
#     print("test...")
#     net.load_state_dict(best_state)
#     res = infer(net, X, G, lbl, test_mask, test=True)
#     print(f"final result: epoch: {best_epoch}")
#     print(res)