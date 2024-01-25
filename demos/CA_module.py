"""
Our own imeplemtation of the functions from the CA paper
"""

import numpy as np
from math import floor
from scipy.special import ellipeinc


def start_grid(gridsize_x, gridsize_y, p,p_1):
    assert gridsize_x % 2 == 1 and gridsize_y % 2 == 1, 'Gridsize must be odd'

    grid = np.zeros((gridsize_x, gridsize_y))
    
    center_x = int((gridsize_x-1)/2)
    center_y = int((gridsize_y-1)/2)

    # Assign nonzero values to individuals outside the grid    
    R = gridsize_x/2
    for x_idx in range(gridsize_x):
        for y_idx in range(gridsize_y):
            # Only individuals inside the circle are considered
            if d(x_idx, y_idx, center_x, center_y) <= R:
                # Assign a value with prob p
                random_number = np.random.rand(1)
                if random_number < p:
                    # Get -1 or 1 with p1
                    random_number = np.random.rand(1)
                    grid[x_idx, y_idx] = 1 if random_number < p_1 else -1
    
    # Add leader in center with opinion 1

    grid[center_x, center_y] = 1
    
    return grid


def get_number_of_nodes_in_grid(grid):
    N = np.count_nonzero(grid)
    return N


def get_node_coordinates(grid):
    N = get_number_of_nodes_in_grid(grid)
    node_coordinates = np.zeros((N, 2))

    n = 0
    for x, y in np.ndindex(grid.shape):
        if grid[x, y] != 0:
            node_coordinates[n, 0] = x
            node_coordinates[n, 1] = y
            # Update the index to save in the right row of node_coordinates
            n += 1
    return node_coordinates


def get_leader_node_index(node_coordinates, gridsize_x, gridsize_y):
    center_x = int((gridsize_x-1)/2)
    center_y = int((gridsize_y-1)/2)

    leader_node_index = 0
    N = node_coordinates.shape[0]

    for n in range(N):
        if node_coordinates[n, 0] == center_x and node_coordinates[n, 1] == center_y:
            leader_node_index = n
    return leader_node_index


def get_beta_matrix(N, beta_people, beta_leader, leader_node_index):
    beta_matrix = np.full(N, beta_people)
    beta_matrix[leader_node_index] = beta_leader
    return beta_matrix


def d(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)


def get_distance_matrix(node_coordinates):
    # Distance matrix (only needed to compute it once!)
    # It is a 3D matrix of size GRIDSIZE_X,GRIDSIZE_,N
    # Each submatrix is the distance grid from each node
    # TODO (Secondary): Optimize this knowing the distance matrix is symetric
    N = node_coordinates.shape[0]
    distance_matrix = np.zeros((N, N))

    # For all nodes
    for n_i in range(N):
        # Compute distance to all other nodes
        for n_j in range(N):
            # Retrieve those values
            x0 = node_coordinates[n_i, 0]
            y0 = node_coordinates[n_i, 1]
            x1 = node_coordinates[n_j, 0]
            y1 = node_coordinates[n_j, 1]

            # Update the distance
            distance_matrix[n_i, n_j] = d(x0, y0, x1, y1)

    # Extra, set closest neighboor distance to 1
    # And other distances to quadratic
    # Iterate rows

    return distance_matrix


def q(mean):
    # Probability distribution for the node influence
    return np.random.uniform(0, 2*mean)

    ## Probability distribution for the node influence
    # TODO: Make it a distribution, was simplified here
    #return 1 #np.random.uniform(0,2*mean)

def get_node_influences(N, mean, leader_node_index, s_L):
    # Influence (computed once!)
    node_influences = np.zeros(N)

    for i in range(N):
        if i == leader_node_index:
            node_influences[i] = s_L
        else:
            node_influences[i] = q(mean)

    return node_influences


def update_opinion(input_grid, N, node_influences, node_coordinates, distance_matrix, leader_node_index, beta_leader, beta_people, temperature, h):
    grid = input_grid.copy() # to ensure we don't modify the input grid
    # Update opinion of each node
    for i in range(N):

        # First compute impact
        # Retrieve node opinion from grid
        # and node influence from matrix
        i_x, i_y = int(node_coordinates[i, 0]), int(node_coordinates[i, 1])

        sigma_i = grid[i_x, i_y]
        s_i = node_influences[i]

        # Compute sum by looping over other nodes
        summation = 0
        for j in range(N):
            if j != i:
                # Retrieve their opinion and influence and distance
                # And add the term
                j_x, j_y = int(node_coordinates[j, 0]), int(
                    node_coordinates[j, 1])

                sigma_j = grid[j_x, j_y]
                s_j = node_influences[j]
                d_ij = distance_matrix[i, j]

                # Compute the function of the distance
                # TODO: Make it a function
                g_d_ij = d_ij

                summation += (s_j * sigma_i * sigma_j)/(g_d_ij)

        # Combine to get social impact
        if i == leader_node_index:
            I_i = -s_i*beta_leader - sigma_i*h - summation
        else:
            I_i = -s_i*beta_people - sigma_i*h - summation

        # Update opinion
        if temperature == 0:
            # Compute probability
            sigma_i = -np.sign(I_i * sigma_i)
        else:
            # Compute probability of change and update if neccesary
            probability_staying = (np.exp(-I_i / temperature)) / \
                (np.exp(-I_i / temperature) + np.exp(I_i / temperature))
            opinion_change = bool(np.random.rand(1) > probability_staying)

            if opinion_change:
                sigma_i = -sigma_i

        # Save the new opinion on the grid
        grid[i_x, i_y] = sigma_i

    # Save the updated grid of the time step
    # simulation[time_step+1,:,:] = grid
    return grid


def analytical_expect_clusters(r, beta, h, s_l):
    # Ensure both solutions are > 0

    print('First half (2*pi*R-sqrt(pi)+beta-h)^2:',
          (2*np.pi*r - np.sqrt(np.pi) + beta - h)**2)
    print('Second half (32*s_l):', 32*s_l)

    condition_1 = bool(
        (2*np.pi*r - np.sqrt(np.pi) + beta - h)**2 - 32*s_l >= 0)
    condition_2 = bool(
        (2*np.pi*r - np.sqrt(np.pi) - beta - h)**2 - 32*s_l >= 0)
    return condition_1 and condition_2


def a(r, betta, h, s_l):
    """
    Calculate the cluster size in determistic model case assuming g(r) = r and mean s = 1
    """
    a_1 = 1/16*(2*np.pi*r - np.sqrt(np.pi) + betta - h +
                np.sqrt((2*np.pi*r - np.sqrt(np.pi) + betta - h)**2 - 32 * s_l))
    a_2 = 1/16*(2*np.pi*r - np.sqrt(np.pi) - betta - h -
                np.sqrt((2*np.pi*r - np.sqrt(np.pi) - betta - h)**2 - 32 * s_l))

    return a_1, a_2

def a_positive(r,betta,h,s_l):
    # + beta + sqrt (stable cluster)
    return 1/16*(2*np.pi*r - np.sqrt(np.pi) + betta - h + np.sqrt((2*np.pi*r - np.sqrt(np.pi) + betta - h)**2 - 32 * s_l))

def a_negative(r,betta,h,s_l):
    # + beta - sqrt (unstable solution)
    return 1/16*(2*np.pi*r - np.sqrt(np.pi) + betta - h - np.sqrt((2*np.pi*r - np.sqrt(np.pi) + betta - h)**2 - 32 * s_l))

def a_stable(r,betta,h,s_l):
    # - beta no sqrt
    return 1/16*(2*np.pi*r - np.sqrt(np.pi) + betta - h)

################################

def minimun_leader_strength(r,beta,h):
    return (2*np.pi*r -np.sqrt(np.pi) -h )/beta

def maximun_leader_strength(r,beta,h):
    # TODO: Add with -beta, but one should be enough
    return (1/32)*(2*np.pi*r -np.sqrt(np.pi) + beta -h )**2

################################


def cluster_size_leader(grid,distance_matrix,leader_node_index,node_coordinates): 
    # Find opinion of leader!
    gridsize_x,gridsize_y = grid.shape
    center_x = int((gridsize_x-1)/2)
    center_y = int((gridsize_y-1)/2)
    leader_opinion = grid[center_x, center_y]

    # Get distance of nodes to leader from distance matrix!!!
    leader_distance_matrix = distance_matrix[leader_node_index,:]
    
    # Cluster has radius r if all nodes at a smaller distance than r
    # to the center have the same opinion as the leader
    # Start with radius 0
    c_radius = 0
    max_c_radius = floor(gridsize_x/2) # TODO: Generalize to rectangle, this assumes square
    consulted_nodes =np.array([])

    while c_radius < max_c_radius:
        # Find all nodes in distance_matrix closer that c_radius
        nodes = np.where(leader_distance_matrix <= c_radius)[0]
        
        #print('Nodes',nodes,consulted_nodes.astype(int))
        #nodes = np.where(nodes != consulted_nodes)[0]

#        print(np.where(nodes != consulted_nodes))
        # Remove consulted nodes from nodes
        #if c_radius>0:
        #    nodes = np.delete(nodes, consulted_nodes.astype(int))
        #print('nodes',nodes[0])

        for n in nodes:
            nx,ny = node_coordinates[n, 0],node_coordinates[n, 1]
            if int(grid[int(nx),int(ny)]) != int(leader_opinion):
                # If somebody has different opinion than leader, then we dont have cluster
                #print('NOOO')
                return c_radius
        
        # TODO: Remove consulted nodes
        #consulted_nodes = np.append(consulted_nodes, nodes, axis=0)
        #print(consulted_nodes)

        c_radius += 1

    # Turn radius to size
    #c_size = np.pi*c_radius**2

    return c_radius#,c_size



# do not know which a to use but use a1 first 
# Inside and outside impact
def Impact_in(s_l,a1,r,c_radius,beta):
    
    return -s_l/d - 8*a1*ellipeinc(c_radius/a1,np.pi/2) + 4*r*ellipeinc(c_radius/r,np.pi/2) + 2*np.sqrt(np.pi) + beta

def Impact_out(s_l,a1,r,c_radius,beta):
    
    return s_l/d + 8*a1*ellipeinc(c_radius/a1,np.arcsin(a1/c_radius)) - 4*r*ellipeinc(c_radius/r,np.pi/2) + 2*np.sqrt(np.pi) + beta
