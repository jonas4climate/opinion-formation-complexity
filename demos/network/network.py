import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import logging
from logging import warning, error, info, debug
from matplotlib.animation import FuncAnimation

logging.basicConfig(level=logging.WARNING)

def longest_opinion_path(G, leader_opinion, node_current, path_current, length_current, visited):
    # Mark node you are at and move one step to the next node with the leader opinion
    visited.add(node_current)
    path_current.append(node_current)

    #Find neighbors with consensus
    consensus_neighbors = [neighbor for neighbor in G.neighbors(node_current) if G.nodes[neighbor]['opinion'] == leader_opinion]

    # Recursively look at the neighbor with the leader opinion
    if consensus_neighbors:
        for neighbor in consensus_neighbors:
            if neighbor not in visited:
                length_current += 1
                longest_opinion_path(G, leader_opinion, neighbor, path_current, length_current, visited)

    return path_current, length_current

def longest_path(G, leader_opinion):
    #Find the longest path of nodes with the same opinion of the leader

    longest_path = []
    longest_length = 0
    visited = set()

    for node in G.nodes:
        if G.nodes[node]['opinion'] == leader_opinion and node not in visited:
            # Start DFS from the current node
            current_path, current_length = longest_opinion_path(G, leader_opinion, node, [], 0, visited)

            # Update the longest path and length if the current path is longer
            if current_length > longest_length:
                longest_path = current_path
                longest_length = current_length

    return longest_path


def grid_distance_metric(node_1, node_2, type='euclidean'):
    x0, y0 = node_1
    x1, y1 = node_2
    if type == 'euclidean':
        return np.sqrt((x1-x0)**2 + (y1-y0)**2)
    elif type == 'von_neumann':
        return abs(x1-x0) + abs(y1-y0)
    elif type == 'moore':
        return max(abs(x1-x0), abs(y1-y0))


def prob_dist_s_people(mean, type='uniform'):

    # Probability distribution for the node influence
    if type == 'uniform':
        return np.random.uniform(0, 2*mean)

    if type == 'normal':
        return np.abs(np.random.normal(mean, scale=1))

    if type == 'exponential':
        return np.random.exponential(mean)


def g(distance_ij, type='linear', c=1):
    """
    Define function g based on distance distance_ij and constant c
    """
    if type == 'linear':
        return c * distance_ij
    elif type == 'exponential':
        return c * np.exp(distance_ij)
    elif type == 'power':
        return distance_ij**c
    elif type == 'logarithmic':
        return c * np.log(distance_ij)

def leader_degree(leader_degree, avg_degree, c=2):
    """
    Returns leader_degree on the criteria that leader degree is minimal factor c as big as the average degree of the network
    """
    if leader_degree <= c * avg_degree:
        return avg_degree * c

    else:
        return leader_degree


class Network(object):

    def __init__(self, gridsize_x, gridsize_y, p_occupation, p_opinion_1, temp, h, beta, beta_leader, s_mean, s_leader, dist_func='euclidean', dist_scaling_func='linear', dist_scaling_factor=1, s_prob_dist_func='uniform', network_type='grid', ba_m=4, neighbor_dist=1, c_leader = 1):
        # Parameters directly provided
        self.gridsize_x, self.gridsize_y = gridsize_x, gridsize_y
        self.p_occupation = p_occupation
        self.p_opinion_1 = p_opinion_1
        self.temp = temp
        self.h = h
        self.beta = beta
        self.beta_leader = beta_leader
        self.s_mean = s_mean
        self.s_leader = s_leader
        self.network_type = network_type
        self.c_leader = c_leader

        # Functions passed
        self.d = lambda node_1, node_2: grid_distance_metric(
            node_1, node_2, type=dist_func)
        self.g = lambda d: g(d, type=dist_scaling_func, c=dist_scaling_factor)
        self.q = lambda mean: prob_dist_s_people(mean, type=s_prob_dist_func)

        # Network itself
        self.G = self.__initialize_network(type=network_type, ba_m=ba_m, neighbor_dist=neighbor_dist)
        self.N = self.G.number_of_nodes()

    def __initialize_network(self, type='grid', ba_m=4, neighbor_dist=1):
        """
        Initialize network based on initial parameter values
        """

        # Create 2D grid graph
        if type == 'grid':
            G = nx.grid_2d_graph(self.gridsize_x, self.gridsize_y)

            # Remove nodes outside of radius R
            center_node = nx.center(G)[0]
            R = self.gridsize_x / 2
            nodes_to_remove = [
                node for node in G.nodes() if self.d(center_node, node) > R]
            G.remove_nodes_from(nodes_to_remove)

            # Create edges between all nodes with attribute distance (shortest path length seen from full grid)
            nodes = list(G.nodes())
            for source in nodes:
                for neighbor in nodes:
                    if source != neighbor:
                        distance = sum(abs(source[i] - neighbor[i])
                                    for i in range(2))
                        G.add_edge(source, neighbor, distance=distance)

            # Remove nodes with probability (1-p)
            p_grid = np.random.rand(self.gridsize_x, self.gridsize_y)
            nodes_to_remove = [(x, y) for x, row in enumerate(p_grid)
                            for y, value in enumerate(row) if value > self.p_occupation]
            G.remove_nodes_from(nodes_to_remove)

            # Ensure the leader is present in the graph
            self.__ensure_leader(G, center_node)

            # Initialize attributes of the graph
            self.__initialize_attributes(G, center_node)

            return G
        
        elif type == 'barabasi-albert':
            R = self.gridsize_x / 2
            N_in_circle = sum(1 for x in range(self.gridsize_x)
                    for y in range(self.gridsize_y) if self.d((x, y), (R, R)) <= R)
            N = int(N_in_circle * self.p_occupation)

            warning(f'Assuming number of nodes is counted accurately as {N}.')
            G = nx.barabasi_albert_graph(N, m=ba_m)
            info(f'Created Barabasi-Albert graph with N={N} and m={ba_m}.')

            # Assume this is the leader because its the most connected node?
            leader_node = max(G.degree, key=lambda x: x[1])[0]
            self.leader_node = leader_node

            # Generate the average degree of all nodes in the network
            degrees = G.degree()
            sum_degrees = sum(dict(G.degree()).values())
            avg_degree = int(sum_degrees / len(degrees))
            self.average_degree = avg_degree

            # Calculate the desired amount of degrees of leader node
            leader_degree_current = G.degree(leader_node)
            c_leader = self.c_leader
            leader_degree_final = leader_degree(leader_degree_current, avg_degree, c_leader)

            edges_to_add = leader_degree_final - leader_degree_current

            # Adding the extra edges
            for _ in range(edges_to_add):
                # Choose a random node to connect to
                target = np.random.choice(list(G.nodes - {leader_node}))

                if not G.has_edge(leader_node, target):
                    G.add_edge(leader_node, target)

            nodes = list(G.nodes())
            for source in nodes:
                for neighbor in G[source]:
                    distance = 1 # TODO: tweak to not always make it 1 but the distance in the grid (additional for loop for neighbor of neighbors?)

                    #If we do this, how do we still measure connectivity? Because still a node is connceted
                    G.add_edge(source, neighbor, distance=distance)

            # Leader is always in the "center" as there is a central hub for BA graphs, so no need to ensure leader is present
                    
            # Initialize attributes of the graph
            self.__initialize_attributes(G, leader_node)

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
                    distance = sum(
                        abs(other_node[i] - center_node[i]) for i in range(2))
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
        opinions = np.random.choice(
            [1, -1], size=N, p=[self.p_opinion_1, 1-self.p_opinion_1])
        influences = self.__get_node_influences(N)
        betas = [self.beta] * N
        impacts = np.zeros(N)

        # Assign attributes directly to nodes
        for node, opinion, influence, beta, impact in zip(G.nodes, opinions, influences, betas, impacts):
            G.nodes[node]['opinion'] = opinion
            G.nodes[node]['influence'] = influence
            G.nodes[node]['beta'] = beta
            G.nodes[node]['impact'] = impact

        # Adjust leader influence
        G.nodes[center_node]['influence'] = self.s_leader
        G.nodes[center_node]['opinion'] = -1

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
        neighbors = list(G_copy.neighbors(target_node))
        for neighbor in neighbors:
            sigma_j = G_copy.nodes[neighbor]['opinion']
            s_j = G_copy.nodes[neighbor]['influence']
            d_ij = G_copy[target_node][neighbor]['distance']

            g_d_ij = self.g(d_ij)
            summation += (s_j * sigma_i * sigma_j) / (g_d_ij)

        impact = -s_i * beta - sigma_i * self.h - summation
        return impact

    def __update_opinion(self, sigma_i, impact):
        """
        Update node opinion sigma_i following the model formula
        """
        if self.temp == 0:
            new_opinion = -np.sign(impact * sigma_i)
            if new_opinion != sigma_i:
                info(f"Opinion change: {sigma_i} -> {new_opinion}")
        else:
            probability_staying = np.exp(-impact / self.temp) / (
                np.exp(-impact / self.temp) + np.exp(impact / self.temp))
            opinion_change = probability_staying < np.random.rand()
            new_opinion = -sigma_i if opinion_change else sigma_i

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
        G_prev_step = self.G.copy()

        for node in nodes:
            # First compute the current impact asserted on the node
            impact = self.__get_impact(node, G_prev_step)
            self.G.nodes[node]['impact'] = impact

            # Update opinion of node
            sigma_i = self.G.nodes[node]['opinion']
            new_opinion = self.__update_opinion(sigma_i, impact)
            self.G.nodes[node]['opinion'] = new_opinion


    def evolve(self, timesteps):
        """
        Evolve the network graph for timesteps
        """

        opinion_history = np.ndarray((timesteps+1, self.N))
        opinions = np.array([data['opinion']
                            for _, data in self.G.nodes(data=True)])
        opinion_history[0] = opinions

        path_history = np.ndarray((timesteps+1),dtype=object)
        path_history[0] = []

        for t in tqdm(range(timesteps)):

            self.update_network()
            opinions = np.array([data['opinion']
                                for _, data in self.G.nodes(data=True)])
            opinion_history[t+1] = opinions

            if self.network_type == 'barabasi-albert':
                longest_path_t = longest_path(self.G, self.G.nodes[self.leader_node]['opinion'])
                path_history[t+1]= longest_path_t

        data = {'opinions': opinion_history, 'longest_path': path_history}
        return data

    def plot_opinion_network_at_time_t(self, data, t, save=False):
        """
        Plot the opinion network graph at timestep t from data returned by `evolve()`
        """
        plt.figure(figsize=(6, 6), layout='tight')
        opinion_history = data['opinions']

        pos = {}
        for n in self.G.nodes:
            a, b = n
            pos[n] = np.array([a, b])

        nx.draw_networkx_nodes(
            self.G, pos, node_color=opinion_history[0], node_size=100, vmin=-1, vmax=1)
        nx.draw_networkx_edges(self.G, pos, alpha=0.1)
        plt.axis("equal")
        plt.grid(False)
        plt.axis(False)
        plt.title(
            f"Opinion network at $t={t}$ ($T={self.temp}$, $s_l={self.s_leader}$, $\\hat{{s}}$={self.s_mean}, $\\beta$={self.beta}, $p_{{occ}}$={self.p_occupation}, $p_{{1}}$={self.p_opinion_1})")
        if save:
            plt.savefig(
                f'figures/{self.gridsize_x}x{self.gridsize_y}_opinion_network_t={t}.png', dpi=300)

    def plot_opinion_network_evolution(self, data, interval=500, save=False, draw_edges=False):
        """
        Plot the evolution of the opinion network graph from data returned by `evolve()`
        """

        plt.figure(figsize=(6, 6), layout='tight')
        opinion_history = data['opinions']

        if self.network_type == 'grid':
            pos = {}
            for n in self.G.nodes:
                a, b = n
                pos[n] = np.array([a, b])

            def update(t):
                plt.clf()
                nx.draw_networkx_nodes(
                    self.G, pos, node_color=opinion_history[t], node_size=100, vmin=-1, vmax=1)
                if draw_edges:
                    nx.draw_networkx_edges(self.G, pos, alpha=0.1)
                plt.axis("equal")
                plt.grid(False)
                plt.axis(False)
                plt.title(
                    f"Opinion network ({self.network_type} type) at $t={t}$ ($T={self.temp}$, $s_l={self.s_leader}$, $\\hat{{s}}$={self.s_mean}, $\\beta$={self.beta}, $p_{{occ}}$={self.p_occupation}, $p_{{1}}$={self.p_opinion_1})", fontsize = 8)
                return plt

        elif self.network_type == 'barabasi-albert':
            path_history = data['longest_path']
            pos = nx.spring_layout(self.G)

            def update(t):
                plt.clf()

                #Place leader as a bigger node at the center of network representation
                leader_position = pos[self.leader_node]
                final_pos = {node: (x - leader_position[0], y - leader_position[1]) for node, (x, y) in
                                pos.items()}
                node_sizes = [100 if node == self.leader_node else 20 for node in self.G.nodes]

                nx.draw_networkx_nodes(self.G, pos, node_color=opinion_history[t], node_size=node_sizes)

                # #Subgraph for the longest path
                # path_edges = [(path_history[t][i], path_history[t][i + 1]) for i in range(len(path_history[t]) - 1)]
                # path_subgraph = self.G.edge_subgraph(path_edges)
                #
                #
                # #Illustrate subgraph
                # nx.draw(path_subgraph, pos,node_size = 0, edge_color='green', width=1)

                plt.annotate(f"Longest path length= {len(path_history[t])}", xy=(0.5, 0.0),xycoords='axes fraction', ha='center', va='center')

                if draw_edges:
                    nx.draw_networkx_edges(self.G, final_pos)

                plt.grid(False)
                plt.axis(False)
                plt.title(
                    f"Opinion network ({self.network_type} type) at $t={t}$ ($T={self.temp}$, $c_l={self.c_leader}$, $<k>$={self.average_degree}, $\\beta$={self.beta}, $p_{{occ}}$={self.p_occupation}, $p_{{1}}$={self.p_opinion_1})",
                    fontsize=8)
                return plt

        anim = FuncAnimation(plt.gcf(), update, frames=range(0, opinion_history.shape[0]), interval=interval)

        if save:
            anim.save(
                f'figures/{self.gridsize_x}x{self.gridsize_y}_opinion_network_evolution.mp4', dpi=300)

