import CA_module as ca
import numpy as np
import matplotlib.pyplot as plt

################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 100
TEMPERATURE = 40
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 1
p = 1
p_1 = 0

INFLUENCE_LEADER = 250   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 15

model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, 'normal', 'exponential')
data = model.evolve(TIMESTEPS)
model.plot_opinion_grid_evolution(data, interval=100)
plt.show()