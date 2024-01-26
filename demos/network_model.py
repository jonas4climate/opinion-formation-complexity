import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import logging
from logging import warning, error, info, debug
from CA_module import prob_dist_influence_people as q

#Importing real network data

    # logging.basicConfig(level=logging.INFO)
    #
    # def graph_from_data(path_to_file):
    #     assert path.exists(path_to_file)
    #     list_edges = np.loadtxt(path_to_file, dtype=int)
    #     fb_graph = nx.from_edgelist(list_edges)
    #     info(f'Graph loaded from {path_to_file}')
    #     return fb_graph
    #
    # def graph_statistics(graph):
    #     n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    #     diameter = nx.diameter(graph)
    #     info(f'Graph has {n_nodes} nodes, {n_edges} edges and diameter {diameter}')
    #
    # graph = graph_from_data('../data/facebook_combined.txt')
    # graph_statistics(graph)
    #

def get_node_influences(N, mean):
    # Influence (computed once!)
    node_influences = np.zeros(N)

    for i in range(N):
        node_influences[i] = q(mean)

    return node_influences

def get_impact(node, G):
    sigma_i = G.nodes[node]['opinion']
    s_i = G.nodes[node]['influence']

    summation = 0
    for neighbor in G.nodes:
        if neighbor != node:
            sigma_j = G.nodes[neighbor]['opinion']
            s_j = G.nodes[neighbor]['influence']
            d_ij = G[node][neighbor]['distance']

            # TODO: Make it a function
            g_d_ij = d_ij

            summation += (s_j * sigma_i * sigma_j) / (g_d_ij)

            print(f"{sigma_j}. {d_ij}")




#Define parameters
GRIDSIZE_X, GRIDSIZE_Y = (5,5)
p = 0.5
mean_s = 1

#Create 2D grid graph
G = nx.grid_2d_graph(GRIDSIZE_X, GRIDSIZE_Y)
nodes = list(G.nodes())

#Create edges between all nodes with attribute distance (shortest path length)
for node in nodes:
    neighbors = G.neighbors(node)
    for neighbor in neighbors:
        G.add_edge(node, neighbor, distance=1)

for source in nodes:
    for target in nodes:
        if source != target:
            distance = sum(abs(source[i] - target[i]) for i in range(2))
            G.add_edge(source, target, distance=distance)

#Remove nodes with probability 1-p
p_grid = np.random.rand(GRIDSIZE_X,GRIDSIZE_Y)
nodes_to_remove = [(x, y) for x, row in enumerate(p_grid > p) for y, value in enumerate(row) if value]
G.remove_nodes_from(nodes_to_remove)


#Create the attributes for the remaining nodes
nodes = list(G.nodes())
n = len(nodes)

#Influence and opinion (opinion now set to -1 as in CA_model)
influences = get_node_influences(n, mean_s)
opinions = [-1 for _ in range(n)]

influences_att = dict(zip(nodes, influences))
opinions_att = dict(zip(nodes, opinions))

nx.set_node_attributes(G, influences_att, name='influence')
nx.set_node_attributes(G, opinions_att, name='opinion')


for node in G:
    get_impact(node,G)

#TODO: Iterate over timesteps to update the sigma of all nodes
