"""
Replicates diagram 1 from paper
"""

import CA_module as ca
import numpy as np
import matplotlib.pyplot as plt

################################

NUMBER_OF_SL_VALUES_TO_TEST = 5
SIMS_PER_SL_VALUE = 2

# TODO: Add points with different innit, so we converge to other stable state
# TODO: Add sims with temperature so we can also get unstable states

################################

GRIDSIZE_X,GRIDSIZE_Y = 25,25
TIMESTEPS = 5
TEMPERATURE = 0
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 0
a_0 = 1 # Size of innitial cluster around leader
INFLUENCE_DISTRIBUTION_MEAN = 1

################################

R = GRIDSIZE_X/2
S_L_min = ca.minimun_leader_strength(R,BETA_PEOPLE,H)
S_L_max = ca.maximun_leader_strength(R,BETA_PEOPLE,H)
cluster_min = ca.a(R,BETA_PEOPLE,H,S_L_min)
cluster_max = ca.a(R,BETA_PEOPLE,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5

################################

#if TEMPERATURE == 0:
#    expect_cluster = ca.analytical_expect_clusters(
#        GRIDSIZE_X/2, BETA_PEOPLE, H, INFLUENCE_LEADER)
#    print('Expect clusters?', expect_cluster)


# Iterate all SL values
SL_values = np.linspace(0,2*S_L_max,NUMBER_OF_SL_VALUES_TO_TEST)

# Get matrices for answers
points_x = np.zeros(NUMBER_OF_SL_VALUES_TO_TEST*SIMS_PER_SL_VALUE)
points_y = np.zeros(NUMBER_OF_SL_VALUES_TO_TEST*SIMS_PER_SL_VALUE)


for index in range(NUMBER_OF_SL_VALUES_TO_TEST):

    INFLUENCE_LEADER = SL_values[index]

    print(f'Sim {index+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    for sim in range(SIMS_PER_SL_VALUE):

        model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
        data = model.evolve(TIMESTEPS)
        #simulation = data['opinions']
        #cluster_sizes = data['cluster_sizes']
        last_cluster_size = data['cluster_sizes'][-1]

        # Save the point
        point_index = index*SIMS_PER_SL_VALUE + sim
        points_x[point_index] = INFLUENCE_LEADER
        points_y[point_index] = last_cluster_size

################################

# Plot diagram
fig, ax = ca.plot_diagram(R,BETA_PEOPLE,H)

# Add points
ax.set_xlim([0,2*S_L_max])
ax.set_ylim([0,int(R)+1])
ax.scatter(points_x, points_y)

# Show
plt.grid()
plt.tight_layout()
plt.show(block=True)


