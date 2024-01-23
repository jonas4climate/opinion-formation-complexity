"""
CA model from paper 'Phase transitions in social impact models of
opinion formation'
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
    for iy, ix in np.ndindex(STARTING_GRID.shape):
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

    Inputs
    x0: ...
    x1: ...
    y0: ...
    y1: ...
    
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
    so 
    """
    return x**2

def I(s,sigma,betta,h):
    """
    Social impact exerted on a particular node
    It is a function 
    """
    return -s*betta - sigma*h - np.sum(1/g(d(1,2,1,1)))

def rule(i):
    """
    Updates the opinion of node i based on its neighboors
    """
    return 0




#########
# TESTS #
#########

# Set global parameters
GRIDSIZE_X,GRIDSIZE_Y = 7,7
p = 0.3  # This value represents likelihood of adding an individual to an space in the grid during innitialization
TIMESTEPS = 5
NEIGHBOURHOOD = 'Moore'
#TODO: APPLY_RULE = CustomRule()
#TODO: APPLY_RULE = lambda n, c,t: cpl.totalistic_rule(n, k=2, rule=126) # This is the core of what we need to modify to match the paper


assert GRIDSIZE_X % 2 != 0, f"Gridsize width should be odd {GRIDSIZE_X}"
assert GRIDSIZE_Y % 2 != 0, f"Gridsize height should be odd {GRIDSIZE_X}"


# Innitialize starting grid
# It is a odd 2D np.array that has a 1 in its center, some -1s around it and the rest 0 (both sides must be odd)
STARTING_GRID = innitialize_grid(p)


## Create and initialize the influence of nodes of our STARTING_GRID
# It is a 2D array of same size, with nodes having positive values from a distribution q
# And the central node has a very high value
INFLUENCE_GRID = innitialize_influence_grid(STARTING_GRID)


print('Starting grid:\n',STARTING_GRID)
print('Influence grid:\n',INFLUENCE_GRID)


 


# Distance function
#distance_test = d(0,0,1,1)
#print(distance_test)

# Social impact

