import CA_module as ca
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

model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, P_OCCUPATION, P_AGREE, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, 'normal', 'exponential')
data = model.evolve(TIMESTEPS)
model.plot_opinion_grid_evolution(data, interval=100)
plt.show()