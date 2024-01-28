import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from CA_module import CA, euclidean_distance, prob_dist_influence_people

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 50
TEMPERATURE = 7
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 0

S_L = 120   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1

ca = CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, S_L, INFLUENCE_DISTRIBUTION_MEAN, euclidean_distance, prob_dist_influence_people)
data = ca.evolve(TIMESTEPS)
ca.plot_opinion_grid_evolution(data, interval=150, save=True)
plt.show()