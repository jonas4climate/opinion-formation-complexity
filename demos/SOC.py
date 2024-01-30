"""
PSEUDOCODE TO OBTAIN POWER LAW OUT OF THE CA
"""

import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt


# Step 0 - Find the critical temperature of the system
## Empirically (like shown in critical_temperature.py) or
## Analitically, using the approximations referenced in the CA paper
# Done in threshold_plot!!!

# Step 1 - Set the system to the critical temperature
critical_temperature = 40

# Step 2 - Set system to the parameters used to find the critical temperature
GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 50
#TEMPERATURE = 0
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 1 # In this scenario everybody believes the leader at start

INFLUENCE_LEADER = 100             # The one 
a_0 = 1
INFLUENCE_DISTRIBUTION_MEAN = 1


# Step 3 - Simulate system at this critical temperature
temperature = critical_temperature
model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=temperature, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=INFLUENCE_LEADER, s_mean=INFLUENCE_DISTRIBUTION_MEAN)
data = model.evolve(TIMESTEPS)

# Step 4 - Plot opinion change to see if clusters do form
for t in range(TIMESTEPS):
    model.plot_opinion_grid_at_time_t(data, T)


# Step 5 - Get cluster sizes from simulation
R = GRIDSIZE_X/2
possible_cluster_sizes = R # As many elements as possible cluster radius: 1,2... up to R
cluster_sizes = np.zeros(R)

# Step 6 - Plot them to see if their sizes follow power law


# Step 7 - Verify it with the package powerlaw

