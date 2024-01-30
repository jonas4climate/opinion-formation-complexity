from network import Network
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
P_OPINION_1 = 1
S_LEADER = 300   # Leader influence
S_MEAN = 1

network = Network(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
data = network.evolve(TIMESTEPS)
network.plot_opinion_network_evolution(data, interval=100)
plt.show()