"""
2D CA Simulation using the CA_module.py

"""

import CA_module as ca

import numpy as np
import cellpylib as cpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################################

GRIDSIZE_X,GRIDSIZE_Y = 25,25
TIMESTEPS = 5
TEMPERATURE = 0
BETA_PEOPLE = 4
BETA_LEADER = 4
H = 0
p = 1  # Inital. likelihood of individual in social space
s_L = 200   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1

################################

if TEMPERATURE == 0:
    expect_cluster = ca.analytical_expect_clusters(GRIDSIZE_X/2,BETA_PEOPLE,H,s_L)
    a1,a2 = ca.a(GRIDSIZE_X/2,BETA_PEOPLE,H,s_L)
    print('Expect clusters?',expect_cluster)
    print('Sizes:',str(round(a1,2)),str(round(a2,2)))

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



# Plot influence grid
# TODO: Make grid out of node_influences
#plt.imshow(node_influences, cmap='seismic', interpolation='nearest')
#plt.colorbar()
#plt.title('Influences')
#plt.show(block=False)

# Colors look up table, so we will always map colors correctly


fig, ax = plt.subplots()

for time_step in range(TIMESTEPS):
    
    # Retrieve plot
    grid_t = simulation[time_step,:,:]

    # Create 2D plot
    # TODO: Normalize colormap based on values -> create LUT

    im = ax.imshow(grid_t, cmap='seismic', interpolation='nearest',vmin=-1, vmax=1)
    

    for n in range(N):
        x,y = int(node_coordinates[n,0]),int(node_coordinates[n,1])
        text = ax.text(y, x, int(grid_t[x, y]), ha="center", va="center", color="w")

    #ax.set_title("Frame: "+str(time_step+1)+'/'+str(TIMESTEPS))
    ax.set_title(f'Frame:{(time_step+1)}/{TIMESTEPS},\nT={TEMPERATURE}, H={H}, B={BETA_PEOPLE}, Bl={BETA_LEADER}, sL={s_L}')

    fig.tight_layout()
    plt.show(block=False)
    
    plt.pause(0.3)
    ax.clear()
    #plt.close()

# Keep last frame until manually closing it
plt.show(block=True)
