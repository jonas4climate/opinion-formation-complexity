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

def innitialize_grid(p):
    """
    Returns a 2D numpy array with a 1 in the center,
    -1 in the other cells (with probability p) and 0 elsewhere
    
    # TODO: Eventually, this probability p should decrease with distance to center
    to better mimic the topology
    """

    assert GRIDSIZE_X % 2 != 0, f"Gridsize width should be odd {GRIDSIZE_X}"
    assert GRIDSIZE_Y % 2 != 0, f"Gridsize height should be odd {GRIDSIZE_X}"

    # Create empty array
    STARTING_GRID = np.zeros((GRIDSIZE_X,GRIDSIZE_Y))

    # Add -1 to it randomly with probability p
    ## Create grid of probabilities
    ## Update the values of our grid that met the probability threshold
    prob_grid = np.random.rand(GRIDSIZE_X,GRIDSIZE_Y)
    STARTING_GRID[prob_grid < p] = -1
    
    # Add the leader in the center, which has a 1
    ## Get the index of the center space of each dimension
    ## And update that coordinate of the STARTING_GRID
    center_x = int((GRIDSIZE_X-1)/2)
    center_y = int((GRIDSIZE_Y-1)/2)
    STARTING_GRID[center_x,center_y] = 1
    
    return STARTING_GRID

def q():
    """
    Returns a random positive number from a distribution.
    As of now it is a uniform distribution within 0 and 1
    
    TODO: Add mean as a parameter
    """


    return np.random.rand(1)

def innitialize_influence_grid(STARTING_GRID):
    """
    Returns a np.array of same size as STARTING_GRID, where each node has a certain influence
    given by the distribution q, and the center value gets a much bigger value by design
    
    TODO: Allow for the node influence to follow any q distribution
    """

    # Create an empty matrix
    INFLUENCE_GRID = np.zeros((GRIDSIZE_X,GRIDSIZE_Y)) #STARTING_GRID

    # Fill it with values
    for ix, iy in np.ndindex(STARTING_GRID.shape):
        # If the corresponding position of STARTING_GRID had -1
        if STARTING_GRID[iy, ix] == -1:
            # Give that influence grid coordinate an influence value
            INFLUENCE_GRID[iy, ix] = q()

    # Put a very high value on the center
    center_value = 100
    center_x = int((GRIDSIZE_X-1)/2)
    center_y = int((GRIDSIZE_Y-1)/2)
    INFLUENCE_GRID[center_x,center_y] = center_value

    return INFLUENCE_GRID


def d(x0,y0,x1,y1):
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


def g(x):
    """
    Increasing function of social distance.
    Used to scale the social impact between nodes
    
    TODO: Use network theory to use an even better function for distance scaling
    """
    return x**2

def I(ix,iy):
    """
    Social impact exerted on a particular node i (with coordinates ix and iy) by the other nodes
    Is is a function of the opinion and influence of our node (si, sigma_i),
    the individual fixation parameter (BETTA), the EXTERNAL_INFLUENCE parameter
    and the opinion and influence of other nodes 
    
    """
    
    # Retrieve the opinion and influence of our node
    s_i = INFLUENCE_GRID[ix,iy]
    sigma_i =  STARTING_GRID[ix,iy]

    # Compute the influence of the other nodes
    ## Notice that when there is no node in a coordinate, STARTING_GRID is 0
    influence_sum = 0

    for jx, jy in np.ndindex(STARTING_GRID.shape):    
        ## If we are not in the position of our node
        if not (ix == jx and iy == jy):
            ## And if we have a cell here (sigma_j != 0 or s_j != 0)
            sigma_j =  STARTING_GRID[jx,jy]
            s_j = INFLUENCE_GRID[jx,jy]
                        
            if s_j != 0:
                ### We need to add another term to the sum
                ### Recall that if there is no cell here, we just add a 0
                ### So we should be ok, plus it should be efficient enough
                influence_sum += (s_j * sigma_i * sigma_j )/d(ix,iy,jx,jy)

    return -s_i*BETTA - sigma_i*EXTERNAL_INFLUENCE - influence_sum #


def get_social_impact_grid(INFLUENCE_GRID):
    """
    Returns a grid with the corresponding social impact that every node has from the others
    """

    # Start with an empty matrix
    SOCIAL_IMPACT_GRID = np.zeros(INFLUENCE_GRID.shape)

    # Loop over values that have nodes and update them
    # with their corresponding social impact
    for ix, iy in np.ndindex(INFLUENCE_GRID.shape):
        if INFLUENCE_GRID[ix,iy] != 0:
            
            impact = I(ix,iy)

            SOCIAL_IMPACT_GRID[ix,iy] = impact

    return SOCIAL_IMPACT_GRID


def rule(old_opinion, I_i, T, deterministic):
    """
    Updates the opinion of node at ix iy based on its neighboyrs
    """

    # If we use the deterministic model, update accordingly
    if deterministic:
        new_opinion = -np.sign(I_i * old_opinion)

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


def get_next_step_grid():
    """
    Returns the array of the next time step by applying the opinion changing rule to all cells

    TODO: Turn this into the rule format of Cellpylib (see https://cellpylib.org/additional.html#custom-rules)
    or, alternatively, develop our own functions for simulation and plotting
    """

    # Start with an empty matrix
    NEW_GRID = np.zeros(STARTING_GRID.shape)
    
    # Loop over values that have nodes and update their opinion
    # with their corresponding result from applying the rule

    for ix, iy in np.ndindex(INFLUENCE_GRID.shape):
        if INFLUENCE_GRID[ix,iy] != 0:
            # Get the old opinion and social influence at the node
            old_opinion = STARTING_GRID[ix,iy] 
            I_i = SOCIAL_IMPACT_GRID[ix,iy] # Or just do I_i I(ix,iy), for SOCIAL_IMPACT_GRID could actually be redundant

            # Apply the rule to get the new opinion
            new_opinion = rule(old_opinion,I_i,TEMPERATURE, DETERMINISTIC)

            # And update it in the matrix
            NEW_GRID[ix,iy] = new_opinion

    return NEW_GRID




#########
# TESTS #
#########

# Set global parameters
GRIDSIZE_X,GRIDSIZE_Y = 3,3
p = 0.4  # This value represents likelihood of adding an individual to an space in the grid during innitialization
TIMESTEPS = 10
NEIGHBOURHOOD = 'Moore'
TEMPERATURE = 100
DETERMINISTIC = False

# Model parameters
BETTA = 1
EXTERNAL_INFLUENCE = 1/2

assert GRIDSIZE_X % 2 != 0, f"Gridsize width should be odd {GRIDSIZE_X}"
assert GRIDSIZE_Y % 2 != 0, f"Gridsize height should be odd {GRIDSIZE_X}"


# Initialize starting grid
## It is a odd 2D np.array that has a 1 in its center, some -1s around it and the rest 0 (both sides must be odd)
STARTING_GRID = innitialize_grid(p)

# Create and initialize the influence of nodes of our STARTING_GRID
## It is a 2D array of same size, with nodes having positive values from a distribution q
## And the central node has a very high value
INFLUENCE_GRID = innitialize_influence_grid(STARTING_GRID)

# Same for the social impact grid
SOCIAL_IMPACT_GRID = get_social_impact_grid(INFLUENCE_GRID)

print('Starting grid:\n',STARTING_GRID)
print('Influence grid:\n',INFLUENCE_GRID)
print('Social impact grid:\n',SOCIAL_IMPACT_GRID)

# Test the CA opinion rule implementaion to see if we can update the system
STARTING_GRID = get_next_step_grid()

print('New grid:\n',STARTING_GRID)


# Deterministic limit case
