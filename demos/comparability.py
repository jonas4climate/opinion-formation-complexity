from network.network import Network
from ca.cellular_automata import CA
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 20
TEMP = 0
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 0.5

#Grid
S_LEADER = 150   # Leader influence
S_MEAN = 1

#Barabasi-Albert
C_LEADER = 100

NETWORK_TYPE = 'grid'

ca_model = CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
ca_data = ca_model.evolve(TIMESTEPS)
ca_model.plot_opinion_grid_evolution(ca_data, interval=250, save=True, filename='comparability_ca.mp4')
plt.show()

network = Network(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN , network_type = NETWORK_TYPE, c_leader = C_LEADER, init_using_ca=ca_model)
data = network.evolve(TIMESTEPS)
network.plot_opinion_network_evolution(data, interval=250, save=True, filename='comparability_network.mp4')
plt.show()