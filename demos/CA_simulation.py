"""
2D CA Simulation using the CA_module.py

"""

import CA_module as ca

import numpy as np
import cellpylib as cpl
import matplotlib.pyplot as plt

################################

GRIDSIZE_X,GRIDSIZE_Y = 15,15
TIMESTEPS = 5
TEMPERATURE = 0
BETA_PEOPLE = 100
BETA_LEADER = 10000
H = 1
p = 1  # Inital. likelihood of individual in social space
s_L = 40000   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1

################################

grid = ca.start_grid(GRIDSIZE_X,GRIDSIZE_Y,p)

N = ca.get_number_of_nodes_in_grid(grid)
node_coordinates = ca.get_node_coordinates(grid)
distance_matrix = ca.get_distance_matrix(node_coordinates)


leader_node_index = ca.get_leader_node_index(node_coordinates,GRIDSIZE_X,GRIDSIZE_Y)
beta = ca.get_beta_matrix(N,BETA_PEOPLE,BETA_LEADER,leader_node_index)
node_influences = ca.get_node_influences(N,INFLUENCE_DISTRIBUTION_MEAN,leader_node_index,s_L)

################################

simulation = np.ndarray((TIMESTEPS,GRIDSIZE_X,GRIDSIZE_Y))

# First step
simulation[0,:,:] = grid

# Next steps
for time_step in range(TIMESTEPS-1):
    grid = ca.update_opinion(grid,N,node_influences,node_coordinates,distance_matrix,leader_node_index,BETA_LEADER,BETA_PEOPLE,TEMPERATURE,H)
    simulation[time_step+1,:,:] = grid

################################

# Plot

#cpl.plot2d_animate(simulation,interval=250)

# Custom plotting
EMPTY_COLOR = (0,0,0)
OPINION_COLOR_1 = (255,0,0)
OPINION_COLOR_2 = (0,0,255)


for time_step in range(TIMESTEPS):
    
    # Retrieve plot
    grid_t = simulation[time_step,:,:]

    # Create 2D plot
    plt.imshow(grid_t, cmap='seismic', interpolation='nearest')
    #plt.colorbar()

    plt.title("Frame: "+str(time_step+1)+'/'+str(TIMESTEPS))
    plt.show(block=False)  
    plt.pause(0.3)
    #plt.close()

# Keep last frame until manually closing it
plt.show(block=True)