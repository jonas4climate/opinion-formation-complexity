"""
PSEUDOCODE TO OBTAIN POWER LAW OUT OF THE CA
"""

import ca.CA_module as ca
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
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 1 # In this scenario everybody believes the leader at start

INFLUENCE_LEADER = 100             # The one 
a_0 = 1
INFLUENCE_DISTRIBUTION_MEAN = 1


# Step 3 - Simulate system at this critical temperature
temperature = critical_temperature
model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, temperature, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
data = model.evolve(TIMESTEPS)

# Step 4 - Plot opinion change to see if clusters do form

# Step 5 - Get cluster sizes from simulation
R = GRIDSIZE_X/2
possible_cluster_sizes = R # As many elements as possible cluster radius: 1,2... up to R
cluster_sizes = np.zeros(R)

# Step 6 - Plot them to see if their sizes follow power law

# Step 7 - Verify it with the package powerlaw

