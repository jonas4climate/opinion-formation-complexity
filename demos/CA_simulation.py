"""
2D CA Simulation using the CA_module.py

"""

import CA_module as ca

import numpy as np
import cellpylib as cpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################################

GRIDSIZE_X,GRIDSIZE_Y = 41,41
TIMESTEPS = 5
TEMPERATURE = 0
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
INFLUENCE_LEADER = 300   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1

################################

if TEMPERATURE == 0:
    expect_cluster = ca.analytical_expect_clusters(
        GRIDSIZE_X/2, BETA_PEOPLE, H, INFLUENCE_LEADER)
    print('Expect clusters?', expect_cluster)

grid = ca.start_grid(GRIDSIZE_X, GRIDSIZE_Y, p)

N = ca.get_number_of_nodes_in_grid(grid)
node_coordinates = ca.get_node_coordinates(grid)
distance_matrix = ca.get_distance_matrix(node_coordinates)


leader_node_index = ca.get_leader_node_index(
    node_coordinates, GRIDSIZE_X, GRIDSIZE_Y)
beta = ca.get_beta_matrix(N, BETA_PEOPLE, BETA_LEADER, leader_node_index)
node_influences = ca.get_node_influences(
    N, INFLUENCE_DISTRIBUTION_MEAN, leader_node_index, INFLUENCE_LEADER)

################################

simulation = np.ndarray((TIMESTEPS, GRIDSIZE_X, GRIDSIZE_Y))
cluster_sizes = np.zeros(TIMESTEPS)

# First step
simulation[0, :, :] = grid
cluster_sizes[0] = ca.cluster_size_leader(grid,distance_matrix,leader_node_index,node_coordinates)


#print('First simulation cluster size')
#print('Cluster size:',cluster_size)

#print(1/0)

# Next steps
for time_step in range(TIMESTEPS-1):

    print('Sim step:',time_step)

    grid = ca.update_opinion(grid, N, node_influences, node_coordinates, distance_matrix,
                             leader_node_index, BETA_LEADER, BETA_PEOPLE, TEMPERATURE, H)

    # TODO: Compute size of cluster around center for all time steps
    cluster_size = ca.cluster_size_leader(grid,distance_matrix,leader_node_index,node_coordinates)
    cluster_sizes[time_step+1] = cluster_size
    simulation[time_step+1, :, :] = grid



################################

################################

# Plot
# cpl.plot2d_animate(simulation,interval=250)


# Plot influence grid
# TODO: Make grid out of node_influences
# plt.imshow(node_influences, cmap='seismic', interpolation='nearest')
# plt.colorbar()
# plt.title('Influences')
# plt.show(block=False)

# Colors look up table, so we will always map colors correctly


fig, ax = plt.subplots()

for time_step in range(TIMESTEPS):

    ## PLOT OPINION EVOLUTION IN TIME
    # Retrieve plot
    grid_t = simulation[time_step, :, :]
    cluster_size = cluster_sizes[time_step]

    # Plot it
    im = ax.imshow(grid_t, cmap='seismic',
                   interpolation='nearest', vmin=-1, vmax=1)
    # Text of opinion
    #for n in range(N):
    #    x, y = int(node_coordinates[n, 0]), int(node_coordinates[n, 1])
    #    text = ax.text(
    #        y, x, str(int(grid_t[x, y])), ha="center", va="center", color="w",fontsize=4)
    ax.set_title(f'Frame:{(time_step+1)}/{TIMESTEPS},\n T={TEMPERATURE}, H={H}, B={BETA_PEOPLE}, Bl={BETA_LEADER}, sL={INFLUENCE_LEADER},c_radius={int(cluster_size)}')
    fig.tight_layout()
    plt.show(block=False)

    plt.pause(0.1)
    ax.clear()
    # plt.close()

# Keep last frame until manually closing it
plt.show(block=True)
