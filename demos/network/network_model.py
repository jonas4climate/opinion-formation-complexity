import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from tqdm import tqdm
import logging
from logging import warning, error, info, debug
from matplotlib.animation import FuncAnimation

logging.basicConfig(level=logging.WARNING)

# from ..ca.CA_module import prob_dist_influence_people as q

def euclidean_distance(node_1, node_2):
    """
    Returns the euclidean distance between two nodes
    """

    x1, y1 = node_1
    x2, y2 = node_2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def prob_dist_influence_people(mean):
    # Probability distribution for the node influence
    return np.random.uniform(0, 2*mean)

class Network(object):

    def __init__(self, gridsize_x, gridsize_y, p_occupation, p_opinion, s_mean, beta_people, temperature, s_l, distance_func=euclidean_distance, influence_prob_dist_func=prob_dist_influence_people):
        self.gridsize_x = gridsize_x
        self.gridsize_y = gridsize_y
        self.p_occupation = p_occupation
        self.p_opinion = p_opinion
        self.s_mean = s_mean
        self.beta_people = beta_people
        self.temperature = temperature
        self.s_l = s_l
        self.d = distance_func
        self.q = influence_prob_dist_func
        self.G = self.__initialize_network()
        self.N = self.G.number_of_nodes()


    def __initialize_network(self):
        """
        Initialize network based on initial parameter values
        """

        # Create 2D grid graph
        G = nx.grid_2d_graph(self.gridsize_x, self.gridsize_y)

        # Remove nodes outside of radius R
        center_node = nx.center(G)[0]                           #assert if center_node has more than 1 value?
        R = self.gridsize_x / 2
        nodes_to_remove = [node for node in G.nodes() if self.d(center_node, node) > R]
        G.remove_nodes_from(nodes_to_remove)

        # Create edges between all nodes with attribute distance (shortest path length seen from full grid)
        nodes = list(G.nodes())
        for source in nodes:
            for target in nodes:
                if source != target:
                    distance = sum(abs(source[i] - target[i]) for i in range(2))
                    G.add_edge(source, target, distance=distance)

        # Remove nodes with probability (1-p)
        p_grid = np.random.rand(self.gridsize_x, self.gridsize_y)
        nodes_to_remove = [(x, y) for x, row in enumerate(p_grid) for y, value in enumerate(row) if value > self.p_occupation]
        G.remove_nodes_from(nodes_to_remove)

        # Ensure the leader is present in the graph
        self.__ensure_leader(G, center_node)

        # Initialize attributes of the graph
        self.__initialize_attributes(G, center_node)

        return G
    
    def __ensure_leader(self, G, center_node):
        """
        Ensures network graph has a leader located at the center of its grid
        """

        nodes = list(G.nodes)

        # Check if leader is in the network, if it is not add it in and connect the appropriate edges
        if not G.has_node(center_node):
            G.add_node(center_node)
            for other_node in nodes:
                if other_node != center_node:
                    distance = sum(abs(other_node[i] - center_node[i]) for i in range(2))
                    G.add_edge(other_node, center_node, distance=distance)

    def __initialize_attributes(self, G, center_node):
        """
        Initialize the attributes of each node in network graph.

        Opinion is either -1 or 1 based on probability p_opinion
        Influence is  calculated based upon the mean influecne mean_s
        Beta is equal to BETA_PEOPLE
        Impact is initialized as 0
        """
        N = G.number_of_nodes()

        # Create the attributes for the nodes
        opinions = np.random.choice([-1, 1], size=N, p=[1 - self.p_opinion, self.p_opinion])
        influences = self.__get_node_influences(N)
        betas = [self.beta_people] * N
        impacts = np.zeros(N)

        # Assign attributes directly to nodes
        for node, opinion, influence, beta, impact in zip(G.nodes, opinions, influences, betas, impacts):
            G.nodes[node]['opinion'] = opinion
            G.nodes[node]['influence'] = influence
            G.nodes[node]['beta'] = beta
            G.nodes[node]['impact'] = impact

        # Adjust leader influence
        G.nodes[center_node]['influence'] = self.s_l


    def __get_node_influences(self, N):
        """
        Compute node influence for N nodes based on specified mean s_mean
        """
        # Influence (computed once!)
        node_influences = np.zeros(N)
        for i in range(N):
            node_influences[i] = self.q(self.s_mean)
        return node_influences


    def __get_impact(self, target_node, G_copy):
        """
        Calculate the impact (I_i) asserted on node target_node by all other nodes in the network graph
        """

        sigma_i = G_copy.nodes[target_node]['opinion']
        s_i = G_copy.nodes[target_node]['influence']
        beta = G_copy.nodes[target_node]['beta']

        summation = 0
        for neighbor in G_copy.nodes:
            if neighbor != target_node:
                sigma_j = G_copy.nodes[neighbor]['opinion']
                s_j = G_copy.nodes[neighbor]['influence']
                d_ij = G_copy[target_node][neighbor]['distance']

                # TODO: Make it a function
                g_d_ij = d_ij
                summation += (s_j * sigma_i * sigma_j) / (g_d_ij)

        impact = -s_i * beta - sigma_i * H - summation
        return impact

    def __update_opinion(self, sigma_i, impact):
        """
        Update node opinion sigma_i following the model formula
        """
        # Update node opinion
        if self.temperature == 0:
            new_opinion = -np.sign(impact * sigma_i)
        else:
            # Compute probability of change and update if necessary
            probability_staying = (np.exp(-impact / self.temperature)) / \
                                (np.exp(-impact / self.temperature) + np.exp(impact / self.temperature))
            opinion_change = np.random.rand() < probability_staying  # Fix the condition

            if opinion_change:
                new_opinion = -sigma_i
            else:
                new_opinion = sigma_i  # If no change, keep the current opinion

        return new_opinion


    def update_network(self):
        """
        Perform one timestep of evolution on the netwerk graph and returns the updated graph
        """

        nodes = list(self.G.nodes())

        # Get and print all node attributes
        node_attributes = self.G.nodes.data()
        for node, attributes in node_attributes:
            info(f"Node {node}, with attributes {attributes}")

        # Copy original network
        G_copy = self.G.copy()

        for node in nodes:
            # First compute the current impact asserted on the node
            impact = self.__get_impact(node, G_copy)
            self.G.nodes[node]['impact'] = impact

            # Update opinion of node
            sigma_i = self.G.nodes[node]['opinion']
            new_opinion = self.__update_opinion(sigma_i, impact)
            self.G.nodes[node]['opinion'] = new_opinion

        self.G = G_copy

    def evolve(self, timesteps):
        """
        Evolve the network graph for timesteps
        """
        opinion_history = np.ndarray((timesteps+1, self.N))
        opinions = np.array([data['opinion'] for _, data in self.G.nodes(data=True)])
        opinion_history[0] = opinions

        for t in tqdm(range(timesteps)):
            self.update_network()
            opinions = np.array([data['opinion'] for _, data in self.G.nodes(data=True)])
            opinion_history[t+1] = opinions
        
        data = {'opinions': opinion_history}
        return data

#Define parameters
GRIDSIZE_X, GRIDSIZE_Y = (9,9)
TIMESTEPS = 5

P_OCCUPATION = 1
P_OPINION = 0.5
H = 1
S_MEAN = 1
BETA_PEOPLE = 1
TEMPERATURE = 0

S_L = 100


# Initialize network
network = Network(GRIDSIZE_X, GRIDSIZE_Y, P_OCCUPATION, P_OPINION, S_MEAN, BETA_PEOPLE, TEMPERATURE, S_L)
data = network.evolve(TIMESTEPS)
opinion_history = data['opinions']

pos = {}
for n in network.G.nodes:
    a,b = n
    pos[n] = np.array([a,b])

nx.draw_networkx_nodes(network.G, pos, node_color=opinion_history[0], cmap=plt.cm.RdYlBu, node_size=100)
nx.draw_networkx_edges(network.G, pos, alpha=0.1)
plt.axis("equal")
plt.grid(False)
plt.show()