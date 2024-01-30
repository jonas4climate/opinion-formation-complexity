import numpy as np
import matplotlib.pyplot as plt
from demos.ca.cellular_automata import CA

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 50
TEMP = 7
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 0

S_L = 120   # Leader influence
S_MEAN = 1

ca = CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_L, s_mean=S_MEAN)
data = ca.evolve(TIMESTEPS)
ca.plot_opinion_grid_evolution(data, interval=150, save=True)
plt.show()