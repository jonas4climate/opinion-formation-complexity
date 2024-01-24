"""
Our own imeplemtation of the functions from the CA paper
"""

import numpy as np

################################

def start_grid(GRIDSIZE_X,GRIDSIZE_Y,p):
    grid = np.zeros((GRIDSIZE_X,GRIDSIZE_Y))
    
    # Add people with opinion -1
    # TODO: Update so they can also have opinion 1
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
    return grid

def get_number_of_nodes_in_grid(grid):
    N = np.count_nonzero(grid)
    return N


def get_node_coordinates(grid):

    N = get_number_of_nodes_in_grid(grid)
    node_coordinates = np.zeros((N,2))

    n = 0
    for x,y in np.ndindex(grid.shape):
        if grid[x,y] != 0:
            node_coordinates[n,0] = x
            node_coordinates[n,1] = y
            # Update the index to save in the right row of node_coordinates
            n += 1   
    return node_coordinates

def get_leader_node_index(node_coordinates,GRIDSIZE_X,GRIDSIZE_Y):
    center_x = int((GRIDSIZE_X-1)/2)
    center_y = int((GRIDSIZE_Y-1)/2)
    
    leader_node_index = 0
    N = node_coordinates.shape[0]

    for n in range(N):
        if node_coordinates[n,0] == center_x and node_coordinates[n,1] == center_y:
            leader_node_index = n
    return leader_node_index

def get_beta_matrix(N,BETA_PEOPLE,BETA_LEADER,leader_node_index):
    beta_matrix = np.ones(N)*BETA_PEOPLE
    beta_matrix[leader_node_index] = BETA_LEADER
    return beta_matrix

def d(x0,y0,x1,y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)


def get_distance_matrix(node_coordinates):
    ## Distance matrix (only needed to compute it once!)
    ## It is a 3D matrix of size GRIDSIZE_X,GRIDSIZE_,N
    ## Each submatrix is the distance grid from each node
    ## TODO (Secondary): Optimize this knowing the distance matrix is symetric
    N = node_coordinates.shape[0]
    distance_matrix = np.zeros((N,N))

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
            distance_matrix[n_i,n_j] = d(x0,y0,x1,y1)
    return distance_matrix

def q(mean):
    ## Probability distribution for the node influence
    return np.random.uniform(0,2*mean)

def get_node_influences(N,INFLUENCE_DISTRIBUTION_MEAN,leader_node_index,s_L):
    ## Influence (computed once!)
    node_influences = np.zeros(N)

    for i in range(N):
        if i == leader_node_index:
            node_influences[i] = s_L
        else:
            node_influences[i] = q(INFLUENCE_DISTRIBUTION_MEAN)
    
    return node_influences

################################

def update_opinion(grid,N,node_influences,node_coordinates,distance_matrix,leader_node_index,BETA_LEADER,BETA_PEOPLE,TEMPERATURE,H):
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
                d_ij = distance_matrix[i,j]
                
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
        if TEMPERATURE == 0:
            # Compute probability
            sigma_i = -np.sign(I_i * sigma_i)
        else:
            # Compute probability of change and update if neccesary
            probability_staying = (np.exp(-I_i * TEMPERATURE))/(np.exp(-I_i * TEMPERATURE) + np.exp(I_i * TEMPERATURE))
            opinion_change = bool(np.random.rand(1) > probability_staying)
            
            if opinion_change:
                sigma_i = -sigma_i

        ## Save the new opinion on the grid
        grid[i_x,i_y] = sigma_i

    ## Save the updated grid of the time step
    #simulation[time_step+1,:,:] = grid
    return grid


################################

