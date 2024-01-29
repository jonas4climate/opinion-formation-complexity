import numpy as np
import matplotlib.pyplot as plt
from network import Network

GRIDSIZE_X, GRIDSIZE_Y = (21, 21)
TIMESTEPS = 20
P_OCCUPATION = 1
P_OPINION_1 = 1
H = 0
S_MEAN = 1
BETA = 1
TEMP = 1
S_LEADER = 150

network = Network(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
data = network.evolve(TIMESTEPS)
network.plot_opinion_network_evolution(data, interval=250)
plt.show()