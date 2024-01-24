import networkx as nx
import numpy as np
import os.path as path
import logging
from logging import warning, error, info, debug

logging.basicConfig(level=logging.INFO)

def graph_from_data(path_to_file):
    assert path.exists(path_to_file)
    list_edges = np.loadtxt(path_to_file, dtype=int)
    fb_graph = nx.from_edgelist(list_edges)
    info(f'Graph loaded from {path_to_file}')
    return fb_graph

def graph_statistics(graph):
    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    diameter = nx.diameter(graph)
    info(f'Graph has {n_nodes} nodes, {n_edges} edges and diameter {diameter}')
         
graph = graph_from_data('../data/facebook_combined.txt')
graph_statistics(graph)