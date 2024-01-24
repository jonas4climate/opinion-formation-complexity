"""
CA model from paper 'Phase transitions in social impact models of
opinion formation'

Next steps
TODO: Test the code 
TODO: Replicate the deterministic limit case (2.2 of paper), including the cluster size, to ensure the logic is sound
TODO: Turn the rule implementation into CellPyLib syntax
TODO: Tune the parameters and replicate other parts of the paper


Extra optimizations
TODO: Because the cells do not move and just change their mind, we should be able to optimize the code
computing distances only once, and removing most matrices to speed up things

"""
import cellpylib as cpl
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
# from numba import njit, jit, prange

def initialize_opinion_grid(p_exists,radius):
    """
    Returns a 2D numpy array with a 1 in the center,
    -1 in the other cells (with probability p) and 0 elsewhere
    
    # TODO: Eventually, this probability p should decrease with distance to center
    to better mimic the topology
    """

    assert GRIDSIZE_X % 2 != 0, f"Gridsize width should be odd {GRIDSIZE_X}"
    assert GRIDSIZE_Y % 2 != 0, f"Gridsize height should be odd {GRIDSIZE_X}"

    starting_opinion_grid = np.zeros((GRIDSIZE_X,GRIDSIZE_Y))

    # Add -1 to the opinion grid randomly with probability p
    # Create grid of probabilities
    # Update the values of our grid that met the probability threshold
    p_grid = np.random.rand(GRIDSIZE_X,GRIDSIZE_Y)
    starting_opinion_grid[p_grid < p_exists] = -1
    
    

    # Add the leader in the center, which has opinion 1
    # Get the index of the center space of each dimension
    # And update that coordinate of the STARTING_GRID
    center_x = int((GRIDSIZE_X-1)/2)
    center_y = int((GRIDSIZE_Y-1)/2)
    starting_opinion_grid[center_x,center_y] = 1
    
    # Exclude any individuals beyond the circle radius
    for x_idx in range(GRIDSIZE_X):
        for y_idx in range(GRIDSIZE_Y):
            if d_social_distance(x_idx,y_idx,center_x,center_y) > radius:
               starting_opinion_grid[x_idx,y_idx] = 0
             
    return starting_opinion_grid

def q(mean):
    """
    Returns a random positive number from a distribution.
    As of now it is a uniform distribution within 0 and 1
    
    TODO: Add more parameters :)
    """


    ## To comply with the deterministic scenario of paper (section 2.2)
    ## We must sample from a distribution with mean 1

    return np.random.uniform(0,2*mean)

def initialize_influence_grid(starting_opinion_grid, mean, leader_influence):
    """
    Returns a np.array of same size as STARTING_GRID, where each node has a certain influence
    given by the distribution q, and the center value gets a much bigger value by design
    
    TODO: Allow for the node influence to follow any q distribution
    """

    # Create an empty matrix
    influence_grid = np.zeros((GRIDSIZE_X,GRIDSIZE_Y)) #STARTING_GRID

    # Fill it with values
    for ix, iy in np.ndindex(starting_opinion_grid.shape):
        # If the corresponding position of STARTING_GRID had -1
        if starting_opinion_grid[iy, ix] == -1:
            # Give that influence grid coordinate an influence value
            influence_grid[iy, ix] = q(mean)

    # Put a very high value on the center
    #center_value = 100
    center_x = int((GRIDSIZE_X-1)/2)
    center_y = int((GRIDSIZE_Y-1)/2)
    influence_grid[center_x,center_y] = leader_influence

    return influence_grid


def d_social_distance(x0,y0,x1,y1):
    """
    Returns the distance between two nodes of the grid given by their x and y indexes

    Example:
    If we have a 3x3 matrix and want to compute the distance between the top left node
    and a node in the center, we would call d(0,0,1,1)=1.414...
    i|-|-
    -|j|-
    -|-|-
    """
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)


def g_distance_scaling(x):
    """
    Increasing function of social distance.
    Used to scale the social impact between nodes
    
    TODO: Use network theory to use an even better function for distance scaling
    """
    return x #**2

def I_social_impact(ix, iy, ext_influence, beta):
    """
    Social impact exerted on a particular node i (with coordinates ix and iy) by the other nodes
    Is is a function of the opinion and influence of our node (si, sigma_i),
    the individual fixation parameter (BETA), the EXTERNAL_INFLUENCE parameter
    and the opinion and influence of other nodes 
    
    """
    
    # Retrieve the opinion and influence of our node
    s_i = influence_grid[ix,iy]
    sigma_i = starting_opinion_grid[ix,iy]

    # Compute the influence of the other nodes
    ## Notice that when there is no node in a coordinate, STARTING_GRID is 0
    influence_sum = 0

    for jx, jy in np.ndindex(starting_opinion_grid.shape):    
        ## If we are not in the position of our node
        if not (ix == jx and iy == jy):
            ## And if we have a cell here (sigma_j != 0 or s_j != 0)
            sigma_j =  starting_opinion_grid[jx,jy]
            s_j = influence_grid[jx,jy]
                        
            if s_j != 0:
                ### We need to add another term to the sum
                ### Recall that if there is no cell here, we just add a 0
                ### So we should be ok, plus it should be efficient enough
                influence_sum += (s_j * sigma_i * sigma_j )/d_social_distance(ix,iy,jx,jy)

    value = -s_i*beta - sigma_i*ext_influence - influence_sum

    return  value #

def get_social_impact_grid(influence_grid, ext_influence, beta):
    """
    Returns a grid with the corresponding social impact that every node has from the others
    """

    # Start with an empty matrix
    social_impact_grid = np.zeros(influence_grid.shape)

    # Loop over values that have nodes and update them
    # with their corresponding social impact
    for ix, iy in np.ndindex(influence_grid.shape):
        if influence_grid[ix,iy] != 0:
            impact = I_social_impact(ix,iy, ext_influence, beta)
            social_impact_grid[ix,iy] = impact

    return social_impact_grid


def rule(old_opinion, I_i, T):
    """
    Updates the opinion of node at ix iy based on its neighboyrs
    """
    if old_opinion == 0:
        return 0

    # If we use the deterministic model, update accordingly
    if T==0:
        new_opinion = -np.sign(I_i * old_opinion)
        #print(old_opinion)
        #new_opinion = np.sign(I_i * old_opinion)
        
        #print('new_opinion:',old_opinion)
        #if old_opinion != old_opinion:
        #    a = 1/0

    else:
        p_keeping_opinion = (np.exp(-I_i/T))/(np.exp(-I_i/T)+np.exp(I_i/T))

        # TODO: Add an asert here to ensure 0<p<1
        #print('p_keeping opinion:',p_keeping_opinion)


        # If a random number between 0 and 1 is under the threshold
        # we stay with the old opinion. Otherwise, we change
        if p_keeping_opinion < np.random.rand(1):
            new_opinion = old_opinion
        else:
            new_opinion = -old_opinion

    return new_opinion

def get_next_step_grid(opinion_grid, temperature, influence_grid, ext_influence, beta):
    """
    Returns the array of the next time step by applying the opinion changing rule to all cells

    TODO: Turn this into the rule format of Cellpylib (see https://cellpylib.org/additional.html#custom-rules)
    or, alternatively, develop our own functions for simulation and plotting
    """

    # Start with an empty matrix
    new_opinion_grid = np.zeros(opinion_grid.shape)

    social_impact_grid = get_social_impact_grid(influence_grid, ext_influence, beta)
    
    # Loop over values that have nodes and update their opinion
    # with their corresponding result from applying the rule

    for ix, iy in np.ndindex(opinion_grid.shape):
        if influence_grid[ix,iy] != 0:
            # Get the old opinion and social influence at the node
            old_opinion = opinion_grid[ix,iy] 
            I_i = social_impact_grid[ix,iy] # Or just do I_i I(ix,iy), for SOCIAL_IMPACT_GRID could actually be redundant

            # Apply the rule to get the new opinion
            new_opinion = rule(old_opinion,I_i,temperature)

            # And update it in the matrix
            new_opinion_grid[ix,iy] = new_opinion

    return new_opinion_grid

def analytical_expect_clusters(r,beta,h,s_l):
    # Ensure both solutions are > 0

    print('First half (2*pi*R-sqrt(pi)+beta-h)^2:',(2*np.pi*r - np.sqrt(np.pi) + beta - h)**2)
    print('Second half (32*s_l):',32*s_l)

    condition_1 = bool((2*np.pi*r - np.sqrt(np.pi) + beta - h)**2 - 32*s_l >= 0)
    condition_2 = bool((2*np.pi*r - np.sqrt(np.pi) - beta - h)**2 - 32*s_l >= 0)
    return condition_1 and condition_2

def a(r,betta,h,s_l):
    """
    Calculate the cluster size in determistic model case assuming g(r) = r and mean s = 1
    """
    a_1 = 1/16*(2*np.pi*r - np.sqrt(np.pi) + betta - h + np.sqrt((2*np.pi*r - np.sqrt(np.pi) + betta - h)**2 - 32 * s_l))
    a_2 = 1/16*(2*np.pi*r - np.sqrt(np.pi) - betta - h - np.sqrt((2*np.pi*r - np.sqrt(np.pi) - betta - h)**2 - 32 * s_l))

    return a_1, a_2

def minimun_leader_strength(r,beta,h):
    return (2*np.pi*r -np.sqrt(np.pi) -h )/beta


#########
# TESTS #
#########

# Set global parameters
GRIDSIZE_X,GRIDSIZE_Y = 9,9
p = 1  # This value represents likelihood of adding an individual to an space in the grid during innitialization
TIMESTEPS = 20
NEIGHBOURHOOD = 'Moore'
TEMPERATURE = 0
RADIUS_SOCIAL_SPACE = GRIDSIZE_X/2

# Model parameters
BETA = 1
EXTERNAL_INFLUENCE = 1

assert GRIDSIZE_X % 2 != 0, f"Gridsize width should be odd {GRIDSIZE_X}"
assert GRIDSIZE_Y % 2 != 0, f"Gridsize height should be odd {GRIDSIZE_X}"


# Initialize starting grid
## It is a odd 2D np.array that has a 1 in its center, some -1s around it and the rest 0 (both sides must be odd)
starting_opinion_grid = initialize_opinion_grid(p,radius=RADIUS_SOCIAL_SPACE)

# Create and initialize the influence of nodes of our STARTING_GRID
## It is a 2D array of same size, with nodes having positive values from a distribution q
## And the central node has a very high value
POPULATION_INFLUENCE_MEAN = 1
LEADER_INFLUENCE = 400
influence_grid = initialize_influence_grid(starting_opinion_grid, mean=POPULATION_INFLUENCE_MEAN, leader_influence=LEADER_INFLUENCE)


# Experiment to test if mean is indeed close to 1
# """
# exp_mean = 0
# runs = 1000
# for i in range(runs):
#     INFLUENCE_GRID = innitialize_influence_grid(STARTING_GRID,MEAN,LEADER_INFLUENCE)
#     exp_mean += np.mean(INFLUENCE_GRID)/runs

# print(exp_mean)
# """

# Same for the social impact grid
social_impact_grid = get_social_impact_grid(influence_grid, ext_influence=EXTERNAL_INFLUENCE, beta=BETA)

#print('Starting grid:\n',STARTING_GRID)
#print('Influence grid:\n',INFLUENCE_GRID)
#print(np.mean(INFLUENCE_GRID))

#print('Social impact grid:\n',SOCIAL_IMPACT_GRID)

# Test the CA opinion rule implementaion to see if we can update the system
#STARTING_GRID = get_next_step_grid()

#print('New grid:\n',STARTING_GRID)



# Try to replicate determinisitc case.
# For this we need to get the points from Fig. 1.


opinion_grid_history = np.ndarray((TIMESTEPS+1,GRIDSIZE_X,GRIDSIZE_Y))
opinion_grid_history[0,:,:] = starting_opinion_grid
expecting_clusters = analytical_expect_clusters(RADIUS_SOCIAL_SPACE,BETA,EXTERNAL_INFLUENCE,LEADER_INFLUENCE)
print('Do we expect clusters with these parameters?', expecting_clusters)




# For loop of simulation
for t in range(TIMESTEPS):
    print(t)
    grid = get_next_step_grid(opinion_grid_history[t,:,:], TEMPERATURE, influence_grid, EXTERNAL_INFLUENCE, BETA)
    opinion_grid_history[t+1,:,:] = grid
    # print(grid)

cpl.plot2d(opinion_grid_history, timestep=0, title='timestep 0')
cpl.plot2d(opinion_grid_history, timestep=1, title='timestep 1')
cpl.plot2d(opinion_grid_history, timestep=2, title='timestep 2')
cpl.plot2d(opinion_grid_history, timestep=3, title='timestep 3')
cpl.plot2d(opinion_grid_history, timestep=4, title='timestep 4')
cpl.plot2d(opinion_grid_history, timestep=5, title='timestep 5')

# cpl.plot2d_animate(opinion_grid_history, 'Opinion Grid history animation', show_grid=True)

# Deterministic limit case

# Plot the animation
# TODO: Understand why cpl plot does not work correctly
#cpl.plot2d(cellular_automaton, timestep=0,title='1')
#cpl.plot2d(cellular_automaton, timestep=1,title='2')
#cpl.plot2d(cellular_automaton, timestep=2,title='3')
#cpl.plot2d(cellular_automaton, timestep=3,title='4')
#cpl.plot2d(cellular_automaton, timestep=4,title='5')
#cpl.plot2d_animate(cellular_automaton,interval=250) # Animation

# plt.matshow(opinion_grid_history[0])
# plt.matshow(opinion_grid_history[4])
# plt.show()



# Deterministic case
## Update q to have mean of 1
## Update g(r) to be r

