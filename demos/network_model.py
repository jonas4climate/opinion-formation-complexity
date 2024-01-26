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

def euclidean_distance(node_1, node_2):
    x1, y1 = node_1
    x2, y2 = node_2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def ensure_leader(G, center_node):
    nodes = list(G.nodes)

    # Make sure leader is in the network, if it is not add it in
    if not G.has_node(center_node):
        G.add_node(center_node)
        for other_node in nodes:
            if other_node != center_node:
                distance = sum(abs(other_node[i] - center_node[i]) for i in range(2))
                G.add_edge(other_node, center_node, distance=distance)

def initialize_attributes(G, p_opinion, mean_s, BETA_PEOPLE, center_node, s_L):
    n = G.number_of_nodes()

    # Create the attributes for the nodes
    # Opinion either -1 or 1 based on probability p_opinion, Influence, Beta, and initialize impact as 0
    opinions = np.random.choice([-1, 1], size=n, p=[1 - p_opinion, p_opinion])
    influences = get_node_influences(n, mean_s)
    betas = get_beta_matrix(n, BETA_PEOPLE)
    impact = np.zeros(n)

    # Assign attributes directly to nodes
    for node, opinion, influence, beta, imp in zip(G.nodes, opinions, influences, betas, impact):
        G.nodes[node]['opinion'] = opinion
        G.nodes[node]['influence'] = influence
        G.nodes[node]['beta'] = beta
        G.nodes[node]['impact'] = imp

    # Adjust leader influence
    G.nodes[center_node]['influence'] = s_L


def initialize_network(gridsize_x, gridsize_y, p_occupation, p_opinion, s_L):
    # Create 2D grid graph
    G = nx.grid_2d_graph(gridsize_x, gridsize_y)

    # Remove nodes outside of radius R
    center_node = nx.center(G)[0]                           #assert if center_node has more than 1 value?
    R = gridsize_x / 2
    nodes_to_remove = [node for node in G.nodes() if euclidean_distance(center_node, node) > R]
    G.remove_nodes_from(nodes_to_remove)

    # Create edges between all nodes with attribute distance (shortest path length seen from full grid)
    nodes = list(G.nodes())
    for source in nodes:
        for target in nodes:
            if source != target:
                distance = sum(abs(source[i] - target[i]) for i in range(2))
                G.add_edge(source, target, distance=distance)

    # Remove nodes with probability (1-p)
    p_grid = np.random.rand(gridsize_x, gridsize_y)
    nodes_to_remove = [(x, y) for x, row in enumerate(p_grid) for y, value in enumerate(row) if value > p_occupation]
    G.remove_nodes_from(nodes_to_remove)

    # Ensure the leader is present in the graph
    ensure_leader(G, center_node)

    # Initialize attributes of the graph
    initialize_attributes(G, p_opinion, mean_s, BETA_PEOPLE, center_node, s_L)
    return G

def get_node_influences(N, mean):
    # Influence (computed once!)
    node_influences = np.zeros(N)
    for i in range(N):
        node_influences[i] = q(mean)
    return node_influences

def get_beta_matrix(N, beta_people):
    beta_matrix = np.full(N, beta_people)
    return beta_matrix

def get_impact(node, G):
    sigma_i = G.nodes[node]['opinion']
    s_i = G.nodes[node]['influence']
    beta = G.nodes[node]['beta']

    summation = 0
    for neighbor in G.nodes:
        if neighbor != node:
            sigma_j = G.nodes[neighbor]['opinion']
            s_j = G.nodes[neighbor]['influence']
            d_ij = G[node][neighbor]['distance']

            # TODO: Make it a function
            g_d_ij = d_ij
            summation += (s_j * sigma_i * sigma_j) / (g_d_ij)

    impact = -s_i * beta - sigma_i * H - summation
    return impact


def update_opinion(sigma_i, impact):
    # Update node opinion
    if temperature == 0:
        new_opinion = -np.sign(impact * sigma_i)
    else:
        # Compute probability of change and update if necessary
        probability_staying = (np.exp(-impact / temperature)) / \
                              (np.exp(-impact / temperature) + np.exp(impact / temperature))
        opinion_change = np.random.rand() < probability_staying  # Fix the condition

        if opinion_change:
            new_opinion = -sigma_i
        else:
            new_opinion = sigma_i  # If no change, keep the current opinion

    return new_opinion


# Update each node in network
def update_network(G):
    nodes = list(G.nodes())

    # Get and print all node attributes
    node_attributes = G.nodes.data()
    for node, attributes in node_attributes:
        print(f"Node {node}: {attributes}")

    # Copy original network
    G_copy = G.copy()

    for node in nodes:
        # First compute the current impact asserted on the node
        impact = get_impact(node, G_copy)
        G.nodes[node]['impact'] = impact

        # Update opinion of node
        sigma_i = G.nodes[node]['opinion']
        new_opinion = update_opinion(sigma_i, impact)
        G.nodes[node]['opinion'] = new_opinion

    return G


#Define parameters
GRIDSIZE_X, GRIDSIZE_Y = (21,21)
TIMESTEPS = 5

P_OCCUPATION = 0.5
P_OPINION = 0.5
H = 1
mean_s = 1
BETA_PEOPLE = 1
temperature = 0

s_L = 100

G = initialize_network(GRIDSIZE_X, GRIDSIZE_Y, P_OCCUPATION, P_OPINION, s_L)
for t in range(TIMESTEPS):
    G = update_network(G)

#
#
# #Visualization
# # Extract node values for visualization
# node_values = [data['impact'] for _, data in G.nodes(data=True)]
#
# # Draw the graph with node colors based on the attribute 'value'
# pos = nx.spring_layout(G)  # Positions nodes using the spring layout algorithm
# nx.draw(G, pos, with_labels=True, node_color=node_values, cmap=plt.cm.RdYlBu, node_size=1000)
#
# # Display the plot
# plt.show()
#
