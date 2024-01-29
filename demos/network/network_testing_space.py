from network_model import Network
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 20
TEMPERATURE = 0
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_AGREE = 1

INFLUENCE_LEADER = 300   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1

network = Network(GRIDSIZE_X, GRIDSIZE_Y, P_OCCUPATION, P_AGREE, INFLUENCE_DISTRIBUTION_MEAN, BETA_PEOPLE, TEMPERATURE, INFLUENCE_LEADER, H, influence_prob_dist_func='normal', distance_scaling_func='exponential')
data = network.evolve(TIMESTEPS)
network.plot_opinion_network_evolution(data, interval=100)
plt.show()