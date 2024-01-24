"""
Alternative implementation
More intuitive and fast
"""

import numpy as np
import cellpylib as cpl
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS
GRIDSIZE_X,GRIDSIZE_Y = 55,55
TIMESTEPS = 5

TEMPERATURE = 0
BETA_PEOPLE = 1
BETA_LEADER = 10

H = 1
p = 0.5  # Inital. likelihood of individual in social space
s_L = 100   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1


 # Flags 
PRINT = False # for printing stuff


# FUNCTIONS
def d(x0,y0,x1,y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)




# PARAMETER VALIDATION
assert GRIDSIZE_X % 2 != 0, f"Gridsize width should be odd {GRIDSIZE_X}"
assert GRIDSIZE_Y % 2 != 0, f"Gridsize height should be odd {GRIDSIZE_X}"
assert p >= 0 and p <= 1, f"Probability should be within [0,1], its {p}"



# INNITIALIZE NODES



## Node location
### Create empty grid
grid = np.zeros((GRIDSIZE_X,GRIDSIZE_Y))

### Fill it with nodes of opposing opinion with probability p
# TODO: Make other nodes have also opinion 1
p_grid = np.random.rand(GRIDSIZE_X,GRIDSIZE_Y)
grid[p_grid < p] = -1
    
### Add leader in center with opinion 1
center_x = int((GRIDSIZE_X-1)/2)
center_y = int((GRIDSIZE_Y-1)/2)
grid[center_x,center_y] = 1

### Exclude individuals outside the circle
R = GRIDSIZE_X/2

for x_idx in range(GRIDSIZE_X):
    for y_idx in range(GRIDSIZE_Y):
        if d(x_idx,y_idx,center_x,center_y) > R:
            grid[x_idx,y_idx] = 0
      

### Get number of nodes
N = np.count_nonzero(grid)

### Get node coordinates
### 2D array for X and Y indeces of nodes
node_coordinates = np.zeros((N,2))

n = 0
for x,y in np.ndindex(grid.shape):
    if grid[x,y] != 0:
        node_coordinates[n,0] = x
        node_coordinates[n,1] = y
        # Update the index to save in the right row of node_coordinates
        n += 1

#print(node_coordinates[0])

### Get the index of the leader node
leader_node_index = 0
for n in range(N):
    if node_coordinates[n,0] == center_x and node_coordinates[n,1] == center_y:
        leader_node_index = n

### Get beta matrix
beta = np.ones(N)*BETA_PEOPLE
beta[leader_node_index] = BETA_LEADER


## Distance matrix (only needed to compute it once!)
## It is a 3D matrix of size GRIDSIZE_X,GRIDSIZE_,N
## Each submatrix is the distance grid from each node
## TODO (Secondary): Optimize this knowing the distance matrix is symetric
node_distances = np.zeros((N,N))

### For all nodes
for n_i in range(N):
    # Compute distance to all other nodes
    for n_j in range(N):
        # Retrieve those values
        x0 = node_coordinates[n_i,0]
        y0 = node_coordinates[n_i,1]
        x1 = node_coordinates[n_j,0]
        y1 = node_coordinates[n_j,1]
        
        # Update the distance
        node_distances[n_i,n_j] = d(x0,y0,x1,y1)





## Probability distribution for the node influence
def q(mean):
    return np.random.uniform(0,2*mean)

## Influence (computed once!)
node_influences = np.zeros(N)

for i in range(N):
    if i == leader_node_index:
        node_influences[i] = s_L
    else:
        node_influences[i] = q(INFLUENCE_DISTRIBUTION_MEAN)



if TEMPERATURE == 0:

    ## Check if clusters are expected
    ## TODO: Understand which beta to use here BETA_PEOPLE or average of all
    condition_1 = bool((2*np.pi*R - np.sqrt(np.pi) + BETA_PEOPLE - H)**2 - 32*s_L >= 0)
    condition_2 = bool((2*np.pi*R - np.sqrt(np.pi) - BETA_PEOPLE - H)**2 - 32*s_L >= 0)
    clusters_expected = bool(condition_1 and condition_2)

    a_1 = 1/16*(2*np.pi*R - np.sqrt(np.pi) + BETA_PEOPLE - H + np.sqrt((2*np.pi*R - np.sqrt(np.pi) + BETA_PEOPLE - H)**2 - 32 * s_L))
    a_2 = 1/16*(2*np.pi*R - np.sqrt(np.pi) - BETA_PEOPLE - H - np.sqrt((2*np.pi*R - np.sqrt(np.pi) - BETA_PEOPLE - H)**2 - 32 * s_L))

    print('\n')
    print('Temperature is 0, system is deterministic')
    print('Are clusters expected?',clusters_expected)
    print('Expected size of clusters:',a_1,a_2)

####################################################


simulation = np.ndarray((TIMESTEPS,GRIDSIZE_X,GRIDSIZE_Y))

# First step ready
simulation[0,:,:] = grid


# SIMULATION! Of other steps
for time_step in range(TIMESTEPS-1):
    
    ## Update opinion of each node
    for i in range(N):
        
        ## First compute impact
        ### Retrieve node opinion from grid
        ### and node influence from matrix
        i_x,i_y = int(node_coordinates[i,0]),int(node_coordinates[i,1])

        sigma_i = grid[i_x,i_y]
        s_i = node_influences[i]

        ### Compute sum by looping over other nodes
        summation = 0
        for j in range(N):
            if j != i:
                ### Retrieve their opinion and influence and distance
                ### And add the term
                j_x,j_y = int(node_coordinates[j,0]),int(node_coordinates[j,1])
                
                sigma_j = grid[i_x,i_y]
                s_j = node_influences[i]
                d_ij = node_distances[i,j]
                
                ### Compute the function of the distance
                ### TODO: Make it a function
                g_d_ij = d_ij
                
                summation += (s_j * sigma_i * sigma_j)/(g_d_ij)


        ### Combine to get social impact
        if i == leader_node_index:
            I_i = -s_i*BETA_LEADER - sigma_i*H - summation
        else:
            I_i = -s_i*BETA_PEOPLE - sigma_i*H - summation

        ## Update opinion
        if TEMPERATURE==0:
            # Compute probability
            sigma_i = -np.sign(I_i * sigma_i)
        else:
            # Compute probability of change and update if neccesary
            probability_staying = (np.exp(-I_i * TEMPERATURE))/(np.exp(-I_i * TEMPERATURE)+np.exp(I_i * TEMPERATURE))
            opinion_change = bool(np.random.rand(1) > probability_staying)
            
            if opinion_change:
                sigma_i = -sigma_i

        ## Save the new opinion on the grid
        grid[i_x,i_y] = sigma_i

    ## Save the updated grid of the time step
    simulation[time_step+1,:,:] = grid

    pass

####################################################

# PLOT EVOLUTION
cpl.plot2d_animate(simulation,interval=250) # Animation


if PRINT:
    print('Nodes:',N)
    print('Coordinates:',node_coordinates)
    print('Distances:',node_distances)
    print('Leader node index:',leader_node_index)
    print('Beta matrix',beta)
    print('Influences:',node_influences)

    ### Print grid
    plt.matshow(grid)
    plt.show()
    ### Print influence of nodes
    plt.plot([i for i in range(N)],node_influences)
    plt.show()