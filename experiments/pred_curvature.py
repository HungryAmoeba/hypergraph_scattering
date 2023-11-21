#from hypg_scattering.models.hyper_scattering_net import HSN
import os 
import sys
import code

# fix this later!
#sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
from hypgs.models.hyper_scattering_net import HSN
from hypgs import DATA_DIR
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
import subprocess
import csv
import json

def gen_hypg_deg_dataset(num_v, num_e, num_graphs, save=False, folder_name = 'hypergraphs'):
    hypergraph_dataset = [dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first') for _ in range(num_graphs)]

    if save:
        save_dir = os.path.join(DATA_DIR,'Synthetic', folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for ind, hg in enumerate(hypergraph_dataset):
            save_name = os.path.join(save_dir, f'{ind:04d}')
            # if not os.path.exists(save_name):
            #     os.mkdir(save_name)
            hg.save(save_name)

    return hypergraph_dataset

def save_individual_hypergraphs(hypergraph_dataset, filename_prefix):
    for idx, hypergraph in enumerate(hypergraph_dataset):
        with open(f"{filename_prefix}_{idx}.ihg.tsv", 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            for hyperedge in hypergraph.e[0]:
                writer.writerow(hyperedge)

def save_collection_hypergraphs(hypergraph_dataset, filename):
    with open(f"{filename}.chg.tsv", 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        for idx, hypergraph in enumerate(hypergraph_dataset):
            for hyperedge in hypergraph.e[0]:
                row = [idx] + list(hyperedge)
                writer.writerow(row)

def convert_to_one_index(hypergraph_dataset):
    # assume the nodes are zero indexed
    for hypergraph in hypergraph_dataset:
        for hyperedge_idx, hyperedge in enumerate(hypergraph.e[0]):
            # Increment each node index in the hyperedge by one
            hypergraph.e[0][hyperedge_idx] = tuple(node + 1 for node in hyperedge)
    return hypergraph_dataset

def convert_to_zero_index(hypergraph_dataset):
    # assume the nodes are one indexed
    for hypergraph in hypergraph_dataset:
        for hyperedge_idx, hyperedge in enumerate(hypergraph.e[0]):
            # Increment each node index in the hyperedge by one
            hypergraph.e[0][hyperedge_idx] = tuple(node - 1 for node in hyperedge)
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
    set_seed(88)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"training on device {device}")

    num_graphs = 25 
    num_graphs_test = 10
    num_v = 10
    num_e = 22 
    # generate graph dataset
    train_folder_name = 'train_hypergraphs'
    test_folder_name = 'test_hypergraphs'
    train_dataset = gen_hypg_deg_dataset(num_v, num_e, num_graphs, save=True, folder_name = train_folder_name)
    #dataset = list(zip(hypergraphs, node_labels, edge_labels))
    file_name_train = os.path.join(DATA_DIR, 'Synthetic','train_dataset')
    save_collection_hypergraphs(convert_to_one_index(train_dataset), file_name_train)

    # Define the command to execute
    run_train_curvature = ("/home/sumry2023_cqx3/orchid/bin/orchid.jl "
            "--aggregation mean "
            "--dispersion UnweightedStar "
            "-i /home/sumry2023_cqx3/hypergraph_scattering/data/Synthetic/train_dataset.chg.tsv "
            "-o /home/sumry2023_cqx3/hypergraph_scattering/data/Synthetic/train_dataset.orc.json")

    # Execute the command
    subprocess.run(run_train_curvature, shell=True)

    test_dataset = gen_hypg_deg_dataset(num_v, num_e, num_graphs_test, save = True, folder_name = test_folder_name)
    #test_hypergraphs, test_node_labels, test_edge_labels = gen_hypg_deg_dataset(num_v, num_e, num_graphs_test)
    #test_dataset = list(zip(test_hypergraphs, test_node_labels, test_edge_labels))
    # put diracs on each node
    file_name_test = os.path.join(DATA_DIR, 'Synthetic', 'test_dataset')
    save_collection_hypergraphs(convert_to_one_index(test_dataset), file_name_test)
    
    # Define the command to execute
    run_test_curvature = ("/home/sumry2023_cqx3/orchid/bin/orchid.jl "
            "--aggregation mean "
            "--dispersion UnweightedStar "
            "-i /home/sumry2023_cqx3/hypergraph_scattering/data/Synthetic/test_dataset.chg.tsv "
            "-o /home/sumry2023_cqx3/hypergraph_scattering/data/Synthetic/test_dataset.orc.json")

    # Execute the command
    subprocess.run(run_test_curvature, shell=True)
    
    # Read the JSON file
    train_json_path = os.path.join(DATA_DIR, 'Synthetic', 'train_dataset.orc.json')
    test_json_path = os.path.join(DATA_DIR, 'Synthetic', 'test_dataset.orc.json')

    with open(train_json_path, 'r') as json_file:
        train_data = json.load(json_file)
    with open(test_json_path, 'r') as json_file:
        test_data = json.load(json_file)
    
    train_labels_edges = torch.zeros(num_graphs,num_e)
    train_labels_nodes = torch.zeros(num_graphs,num_v)
    test_labels_edges = torch.zeros(num_graphs_test, num_e)
    test_labels_nodes = torch.zeros(num_graphs_test, num_v)

    for ind in range(num_graphs):
        train_labels_edges[ind, :] = torch.tensor(train_data[ind]['edge_curvature'])
        train_labels_nodes[ind, :] = torch.tensor(train_data[ind]['node_curvature_neighborhood'])
        
    for ind in range(num_graphs_test):
        test_labels_edges[ind, :] = torch.tensor(test_data[ind]['edge_curvature'])
        test_labels_nodes[ind, :] = torch.tensor(test_data[ind]['node_curvature_neighborhood'])

    #import pdb; pdb.set_trace()

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
    # train simple model
    for epoch in trange(25):
        #random.shuffle(train_dataset)
        epoch_loss = 0
        for ind,hg in enumerate(train_dataset):
            #import pdb; pdb.set_trace()
            hg = hg.to(device)
            # is the issue in this step?
            #hg = dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first').to(device)
            #hg = hg.to(device)
            node_label = train_labels_nodes[ind].to(device)
            edge_label = train_labels_edges[ind].to(device)
            #node_label = hg.D_v.values().to(device)
            #edge_label = hg.D_e.values().to(device)

            # initialize starting edge features to be a matrix of zeros
            num_e_hg = hg.num_e 
            Y = torch.zeros(num_e_hg, num_v).to(device)

            net.train()

            st = time.time()
            optimizer.zero_grad()
            node_pred, edge_pred = net(hg, starting_features, Y)
            loss = criterion(torch.squeeze(edge_pred), edge_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_losses.append(loss.item())

            #epoch_loss += train_node_degree(net, hg, starting_features, node_label, optimizer, criterion, epoch)
            #print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")

        #if epoch%10 == 0:
        print(f'Loss on epoch {epoch} is {epoch_loss/len(train_dataset)}')
    
    test_loss = 0
    for ind,hg in enumerate(test_dataset):
        hg = hg.to(device)
        #node_label = hg.D_v.values().to(device)
        #edge_label = hg.D_e.values().to(device)
        #edge_label = edge_label.to(device)
        node_label = test_labels_nodes[ind].to(device)
        edge_label = test_labels_edges[ind].to(device)
        torch.no_grad()
        net.eval()
        num_e_hg = hg.num_e 
        Y = torch.zeros(num_e_hg, num_v).to(device)
        node_pred, edge_pred = net(hg, starting_features, Y) 
        #import pdb; pdb.set_trace()
        loss = criterion(torch.squeeze(edge_pred), edge_label)
        test_loss += loss.item() 
    #import pdb; pdb.set_trace()


    print(net)
    print(f'nonlin is {net.activation}')
    print(f'Test loss is {test_loss/len(test_dataset)}')
    plt.plot(train_losses)
    plt.title("train losses on hyperedge curvature (individual graphs)")
    plt.savefig('figures/train_losses_edge_curve.png')
    code.interact(local=locals())


    # HSN gets .01625 MSE on the test set
        


    


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