import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import logging
from logging import warning, error, info, debug

#from ca.CA_module import prob_dist_influence_people as q # Commented as cant import from neighboor folder

def q(mean):
    # Probability distribution for the node influence
    return np.random.uniform(0, 2*mean)



from matplotlib.animation import FuncAnimation

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
    """
    Returns the euclidean distance between two nodes
    """

    x1, y1 = node_1
    x2, y2 = node_2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def ensure_leader(graph, center_node):
    """
    Ensures network graph has a leader located at the center of its grid
    """

    nodes = list(graph.nodes)

    # Check if leader is in the network, if it is not add it in and connect the appropriate edges
    if not graph.has_node(center_node):
        graph.add_node(center_node)
        for other_node in nodes:
            if other_node != center_node:
                distance = sum(abs(other_node[i] - center_node[i]) for i in range(2))
                graph.add_edge(other_node, center_node, distance=distance)

def initialize_attributes(graph, center_node, p_opinion, mean_s, beta_people, s_L):
    """
    Initialize the attributes of each node in network graph.

    Opinion is either -1 or 1 based on probability p_opinion
    Influence is  calculated based upon the mean influecne mean_s
    Beta is equal to BETA_PEOPLE
    Impact is initialized as 0
    """
    n = graph.number_of_nodes()

    # Create the attributes for the nodes
    opinions = np.random.choice([-1, 1], size=n, p=[1 - p_opinion, p_opinion])
    influences = get_node_influences(n, mean_s)
    betas = [beta_people] * n
    impacts = np.zeros(n)

    # Assign attributes directly to nodes
    for node, opinion, influence, beta, impact in zip(graph.nodes, opinions, influences, betas, impacts):
        graph.nodes[node]['opinion'] = opinion
        graph.nodes[node]['influence'] = influence
        graph.nodes[node]['beta'] = beta
        graph.nodes[node]['impact'] = impact

    # Adjust leader influence
    graph.nodes[center_node]['influence'] = s_L


def initialize_network(gridsize_x, gridsize_y, p_occupation, p_opinion, mean_s, beta_people, s_L):
    """
    Initialize network based on initial parameter values
    """

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
    initialize_attributes(G, center_node, p_opinion, mean_s, beta_people, s_L)
    return G

def get_node_influences(N, s_mean):
    """
    Compute node influence for N nodes based on specified mean s_mean
    """
    # Influence (computed once!)
    node_influences = np.zeros(N)
    for i in range(N):
        node_influences[i] = q(s_mean)
    return node_influences


def get_impact(target_node, graph):
    """
    Calculate the impact (I_i) asserted on node target_node by all other nodes in the network graph
    """

    sigma_i = graph.nodes[target_node]['opinion']
    s_i = graph.nodes[target_node]['influence']
    beta = graph.nodes[target_node]['beta']

    summation = 0
    for neighbor in graph.nodes:
        if neighbor != target_node:
            sigma_j = graph.nodes[neighbor]['opinion']
            s_j = graph.nodes[neighbor]['influence']
            d_ij = graph[target_node][neighbor]['distance']

            # TODO: Make it a function
            g_d_ij = d_ij
            summation += (s_j * sigma_i * sigma_j) / (g_d_ij)

    impact = -s_i * beta - sigma_i * H - summation
    return impact

def update_opinion(sigma_i, impact):
    """
    Update node opinion sigma_i following the model formula
    """
    # Update node opinion
    if TEMPERATURE == 0:
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


def update_network(graph):
    """
    Perform one timestep of evolution on the netwerk graph and returns the updated graph
    """

    nodes = list(graph.nodes())

    # Get and print all node attributes
    node_attributes = graph.nodes.data()
    for node, attributes in node_attributes:
        print(f"Node {node}, with attributes {attributes}")

    # Copy original network
    graph_copy = graph.copy()

    for node in nodes:

        # First compute the current impact asserted on the node
        impact = get_impact(node, graph_copy)
        graph.nodes[node]['impact'] = impact

        # Update opinion of node
        sigma_i = graph.nodes[node]['opinion']
        new_opinion = update_opinion(sigma_i, impact)
        graph.nodes[node]['opinion'] = new_opinion

    return graph


#Define parameters
GRIDSIZE_X, GRIDSIZE_Y = (5,5)
TIMESTEPS = 5

P_OCCUPATION = 1
P_OPINION = 0.5
H = 1
MEAN_S = 1
BETA_PEOPLE = 1
TEMPERATURE = 0

S_L = 100


#Initialize network
G = initialize_network(GRIDSIZE_X, GRIDSIZE_Y, P_OCCUPATION, P_OPINION, MEAN_S, BETA_PEOPLE, S_L)

#Perform iterations
for t in range(TIMESTEPS):
    print(f"Timestep {t}:")
    update_network(G)

    # Plot network evolving over time


# TODO: Draw network in 2D space correctly
print(G)
#print(G.nodes)
#print(G.nodes(data=True))



# #Visualization
# # Extract node values for visualization
node_values = [data['impact'] for _, data in G.nodes(data=True)]
#
# # Draw the graph with node colors based on the attribute 'value'
#pos = nx.spring_layout(G)  # Positions nodes using the spring layout algorithm


pos = {}

for n in G.nodes:
    a,b = n
    pos[n] = np.array([a,b])

#pos = nx.circular_layout(G, scale=1, center=(2,2), dim=2)


#nx.draw_networkx_nodes(G, pos=pos)
#nx.draw_networkx_edges(G, pos=pos)
#nx.draw_networkx_labels(G, pos=pos)

nx.draw(G, pos, with_labels=True, node_color=node_values, cmap=plt.cm.RdYlBu, node_size=100)

#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()


plt.axis("equal")
plt.grid()
# # Display the plot
plt.show()
#
