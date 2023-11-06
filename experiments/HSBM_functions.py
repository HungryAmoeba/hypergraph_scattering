#!/usr/bin/env python
# coding: utf-8

# ## Hypergraph Scattering
# The goal of this program is to generate random hypergraphs and then perform hypergraph scattering

# In[24]:


import random
import networkx as nx
import matplotlib.pyplot as plt
import math
from itertools import combinations, chain
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ### Build basis SBM
# `basic_sbm` builds a basic Stochastic Block Model (SBM) with the following inputs
# * `num_membs` is the number of **total** members in **all** communities
# * `num_commun` is the number of communities
# * `p` is the probability that two members in the **same** community are connected 
# * `q` is the probability that two members in **different** communities are connected 
# 
# `basic_sbm` outputs a graph that is a networkx object

# In[2]:


def basic_sbm(num_memb = 100, num_commun = 2, p = 0.8, q = 0.01):
    # build list sizes, this will put the same number of members into
    # the number of communities `num_commun`
    
    if num_memb % 2 == 0:
        sizes = [num_memb // num_commun] * num_commun
        
    else:
        sizes = [num_memb // num_commun] * num_commun
        sizes[0] = sizes[0] + 1
        
    # build 2d array of probabilities
    
    probs = np.zeros((num_commun, num_commun))
    
    for i in range(num_commun):
        for j in range(num_commun):
            if i == j:
                probs[i][j] = p
                
            else:
                probs[i][j] = q
                
    # create SBM graph G
    
    G = nx.stochastic_block_model(sizes, probs, seed = 0)
    
    return G


# ### Draw basic SBM
# `draw_graph` draws a basic SBM graph with a networkx object `graph`

# In[3]:


def draw_graph(graph):
    num_members = len(graph.nodes())
    colors = []

    if num_members % 2 == 0:
        size = num_members // 2
        colors = (['magenta'] * size) + (['orange'] * size)
    else:
        size = num_members // 2
        colors = (['magenta'] * (size + 1)) + (['orange'] * size)
        
    position = nx.spring_layout(graph)
    
    nx.draw_networkx(graph, 
                     pos = position,
                     with_labels = False,
                     node_size = 175,
                     node_color = colors,
                     alpha = 0.9,
                     width = 0.6,
                     edge_color = 'lightgrey')
    plt.axis('off')
    plt.show()


# ### Draw the block adjacency matrix
# `draw_adj_matrix` plots the adjacency matrix. This visual will show how that members in the same community are more connected than members in different communities

# In[4]:


def draw_adj_matrix(graph):
    adj = nx.to_numpy_array(graph)
    plt.imshow(adj, cmap = plt.cm.get_cmap('binary'))
    plt.title('Adjaceny Matrix of Basic SBM')
    plt.tick_params(top = True, labeltop = True, bottom = False, labelbottom = False)
    plt.show()


# ### Potential triangles to triangles
# `triangle` looks through all potential 3-clique expansions and will become a 3-clique expansion with probability $\alpha$ and will remain a 2-clique with probability $1 - \alpha$.
# 
# This function returns a new `edge_lst` with edges that contain more than 2 members

# In[5]:


def triangle(graph, alpha):
   # need node and edge list
    
    node_lst = list(graph.nodes())
    edge_lst = list(graph.edges())
    
    # want to find all possible combinations of 3 nodes
    
    three_combo = list(combinations(node_lst, 3))
    
    # now, with probability alpha, we will accept a possible
    # three_comb and 1 - alpha, we will reject a possible three comb
    
    edge_copy = edge_lst.copy() # keep a copy of edge_lst
    now_three = 0
    
    for i, j, k in three_combo:
        if ((i, j) in edge_lst) and ((i, k) in edge_lst) and ((j, k) in edge_lst):
            # simualte random uniform var
            rand = np.random.uniform(low = 0, high = 1)
            
            # here, we will aceept if rand is greater than alpha (getting a heads) and 
            # we then remove previous pairwise edges and add new three_combo edge

            if rand <= alpha: # getting a head
                now_three += 1
                
                edge_lst.remove
                ((i, j)) 
                edge_lst.remove((i, k))
                edge_lst.remove((j, k))
                
                edge_lst.append((i, j, k))
    
    # print(f'Previous size of edge_lst: {len(edge_copy)}, New size of edge_lst: {len(edge_lst)}, Number of pairwise edges removed: {now_three}' )
    
    return edge_lst


# ### Generating Random Hypergraphs
# `generate_classes` generates an equal number of random hypergraphs with a large $\alpha$ value and a small $\alpha$ value. Its inputs are
# * `num_graphs` which is the number of how many **total** graphs will be generated
# * `num_membs` is the number of **total** members in **all** communities
# * `num_commun` is the number of communities
# * `p` is the probability that two members in the **same** community are connected 
# * `q` is the probability that two members in **different** communities are connected 
# * `l_alpha` is the large alpha probability
# * `s_alpha` is the small alpha probability
# 
# This function returns two a tuple of lists `(l_alpha_lst, s_alpha_lst)` where each respective list is a list of all randomly generated hypergraphs with a large or small alpha value

# In[6]:


def generate_classes(num_graphs, num_memb, num_commun = 2, p = 0.8, q = 0.01, l_alpha = 0.9, s_alpha = 0.1):
    all_graphs = [] # will hold all simulated SBM graphs
    
    for i in range(num_graphs):
        add_graph = basic_sbm(num_memb, num_commun, p, q)
        all_graphs.append(add_graph)
        
    # the task requires two equal sized classes where one has a large alpha
    # and the other has a small alpha
    
    l_alpha_lst = []
    s_alpha_lst = []

    for _ in range(0, num_graphs // 2):
        large_alpha_graph = triangle(all_graphs[i], l_alpha)
        l_alpha_lst.append(large_alpha_graph) # [1, 0] is the ground truth label for a large alpha value 
        
    for _ in range(num_graphs // 2, num_graphs):
        small_alpha_graph = triangle(all_graphs[i], s_alpha) 
        s_alpha_lst.append(small_alpha_graph) # [0, 1] is the ground truth label for a small alpha value
    
    return (l_alpha_lst, s_alpha_lst)


# ### Build Necessary Hypergraph Representation for PyG
# `build_hgraph` builds the necessary hypergraph representation to be able to use pytorch geometric. It takes in the list of graphs generated from a large and small alpha value. It returns 2 lists and 2 dictionaries
# * `l_alpha_hgraph` and `s_alpha_hgraph` are lists of all hypergraph representations for large and small alpha values repectively
# * `l_alpha_hgraph_dict` and `s_alpha_hgraph_dict` are dictionaries where the keys represent the n-th hyperedge and the value is the list of nodes in that hyperedge

# In[7]:


def build_hgraph(l_alpha_lst, s_alpha_lst):
    l_alpha_hgraph = []
    s_alpha_hgraph = []
    l_alpha_hgraph_dict = {}
    s_alpha_hgraph_dict = {}
    
    for i in range(len(l_alpha_lst)):
        vals = l_alpha_lst[i]
        keys = list(range(len(vals)))
        dct = dict(list(zip(keys, vals)))
        l_alpha_hgraph_dict[i] = dct
        
        lst = []
        lst_indices = []

        for i in range(len(vals)):
            lst.append(vals[i])
            for val in vals[i]:
                lst_indices.append(i)

        flat_lst = list(chain(*lst))
        hyperedge_index = np.array([flat_lst, lst_indices])
        
        l_alpha_hgraph.append(hyperedge_index)
        
    for i in range(len(s_alpha_lst)):
        vals = s_alpha_lst[i]
        keys = list(range(len(vals)))
        dct = dict(list(zip(keys, vals)))
        s_alpha_hgraph_dict[i] = dct
        
        lst = []
        lst_indices = []

        for i in range(len(vals)):
            lst.append(vals[i])
            for val in vals[i]:
                lst_indices.append(i)

        flat_lst = list(chain(*lst))
        hyperedge_index = np.array([flat_lst, lst_indices])
        
        s_alpha_hgraph.append(hyperedge_index)
        
    return (l_alpha_hgraph, l_alpha_hgraph_dict, s_alpha_hgraph, s_alpha_hgraph_dict)


# ### Building Random Signals
# `rand_signals` generates random dirac signals on the random hypergraphs. It does this through randomly selecting `k` nodes and placing a dirac on that node for `n` iterations with replacement.
# 
# `rand_signals` returns a sorted dictionary with all signals

# In[8]:


def rand_signals(num_membs, k, n):
    signals = {}
    
    for _ in range(n):
        verts = random.choices(list(range(num_membs)), k = k)
        
        for v in verts:
            # rand_idx = random.choice(list(range(num_membs)))
            dirac = np.zeros(shape = (num_membs, 1))
            dirac[v] = 1
            signals[v] = dirac
       
    final_signal = sum(signals.values())
    
    return final_signal


# ### Get the Incidence Matrix
# Now that we have hypergraphs, we need to find the incidence matrix. There is no attribute for incidence matrix with networkx objects, so `get_incidence` returns the incidence matrix for a set of hyperedges. 
# 
# This function inputs are
# * `hedge_dict` is a dictionary, you should use the dictionary from `build_hgraph`
# * `num_nodes` is an integer for the number of nodes in a hypergraph

# In[9]:


def get_incidence(hedge_dict, num_nodes):
    num_hedge = len(hedge_dict)
    incidence = np.zeros(shape = (num_nodes, num_hedge))

    for val in hedge_dict.items():
        # print(val)
        hedge, node = val

        for idx in node:
            # print(idx)
            if incidence[idx][hedge] == 0:
                incidence[idx][hedge] = 1

    return incidence


# ### Get Random Walk Matrix
# `get_P` has the same inputs as the previous function and will create the necessary random walk operator to perform scattering

# In[10]:


def get_P(hedge_dict, num_nodes):
    #incidence matrix
    H = get_incidence(hedge_dict, num_nodes)

    # vertex degree matrix (sum rows of H)
    D_v = np.matrix(np.diag(np.array(H).sum(axis = 1)))

    # edge degree matrix (sum cols of H)
    D_e = np.matrix(np.diag(np.array(H).sum(axis = 0)))

    # random walk operator matrix
    P = D_v.I * H * D_e.I * H.T

    N = P.shape[0]

    # return transpose so it operates Px
    return P.T


# ### Get the Wavelet Matrix
# `get_wavelet_matrix` takes a `largest_scale` and returns a list of wavelet
# matrices up to those scales. 
# 
# For example, for largest_scale = 4,
# this will return (Phi_1, Phi 2, Phi_3, Phi_4)

# In[11]:


def get_wavelet_matrix(G, largest_scale, num_nodes):
    P = get_P(G, num_nodes)    # lazy random walk matrix
    N = P.shape[0]  # number of nodes
    powered_P = P   # we will exploit the dyadic scales for computational efficiency
    Phi = powered_P @ (np.identity(N) - powered_P)    # First wavelet
    wavelets = list()
    wavelets.append(Phi)
    
    for scale in range(2, largest_scale + 1):
        powered_P = powered_P @ powered_P               # Returns P^{2^(scale -1)}
        Phi = powered_P @ (np.identity(N) - powered_P)  # Calculate next wavelet
        wavelets.append(Phi)

    return wavelets


# ### Compute Scattering Coefficients
# `geom_scattering` computes the scattering coefficients necessary to perfom classifcation tasks on the random graphs. 

# In[12]:


def geom_scattering(G, s, num_nodes, largest_scale, highest_moment):
    # J = largest_scale, Q = highest_moment, s = signal
    wavelets = get_wavelet_matrix(G, largest_scale, num_nodes)
    scattering_coefficients = []

    # Calculate zero order scattering coefficients
    for q in range(1, highest_moment + 1):
        coeff = np.sum(np.power(s, q))
        scattering_coefficients.append(coeff)

    # Calculate first order scattering coefficients
    for scale in range(largest_scale):
        wavelet_transformed = wavelets[scale] @ s
        abs_wavelet = np.abs(wavelet_transformed)
        
        for q in range(1, highest_moment + 1):
            coeff = np.sum( np.power(abs_wavelet, q) )
            scattering_coefficients.append(coeff)

    # Calculate second order scattering coefficients
    for scale1 in range(1, largest_scale):
        
        # the second scale only goes up to the size of the first scale
        for scale2 in range(scale1):
            first_wavelet_transform = np.abs(wavelets[scale2] @ s )
            second_wavelet_transform = np.abs(wavelets[scale1] @ first_wavelet_transform)
            
            for q in range(1, highest_moment + 1):
                coeff = np.sum(np.power(second_wavelet_transform, q) )
                scattering_coefficients.append(coeff)
    
    return np.array(scattering_coefficients)


# ### All together now! 

# In[20]:


def hg_scattering(num_graphs, num_membs, p, q, l_alpha, s_alpha, k, n, largest_scale, highest_moment):
    l, s = generate_classes(num_graphs = num_graphs, num_memb = num_membs, p = p, q = q, l_alpha = l_alpha, s_alpha = s_alpha)
    
    l_h, l_h_d, s_h, s_h_d = build_hgraph(l, s)
    
    rand_signals_l = [rand_signals(num_membs, k, n) for _ in range(num_graphs // 2)]
    rand_signals_s = [rand_signals(num_membs, k, n) for _ in range(num_graphs // 2)]
    
    geom_l = [[geom_scattering(l_h_d[i], rand_signals_l[i], num_membs, largest_scale, highest_moment), 0] for i in range(num_graphs // 2)]
    geom_s = [[geom_scattering(s_h_d[i], rand_signals_s[i], num_membs, largest_scale, highest_moment), 1] for i in range(num_graphs // 2)]
    
    scat_coeff = [(geom_l + geom_s)[i][0] for i in range(len(geom_l + geom_s))]
    ground_truth = [(geom_l + geom_s)[i][1] for i in range(len(geom_l + geom_s))]
    
    return scat_coeff, ground_truth


# In[21]:


sc, gt = hg_scattering(num_graphs = 50, num_membs = 100, p = 0.5, q = 0.5, l_alpha = 0.9, s_alpha = 0.1, k = 5, n = 5, largest_scale = 3, highest_moment = 2)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(sc, gt, train_size = 0.25, shuffle = True)


# In[26]:


model = linear_model.LogisticRegression()
m = model.fit(X_train, y_train)
y_pred = m.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
print(f'\nAccuracy: {acc}\nPrecision: {prec}')


# In[ ]:




