"""
Basic CA code to ensure 2D CA can be run by everybody
Taken from documentation: https://cellpylib.org/twodim.html

"""

import cellpylib as cpl


# Global parameters
GRIDSIZE_X,GRIDSIZE_Y = 30,30
TIMESTEPS = 10
NEIGHBOURHOOD = 'Moore'
APPLY_RULE = lambda n, c,t: cpl.totalistic_rule(n, k=2, rule=126) # This is the core of what we need to modify to match the paper
STARTING_GRID = cpl.init_simple2d(GRIDSIZE_X, GRIDSIZE_Y) # This should be a numpy.ndarray, and should also be modified to match the paper


# Evolve the CA for the TIMESTEPS
cellular_automaton = cpl.evolve2d(cellular_automaton=STARTING_GRID,
                                  timesteps=TIMESTEPS,
                                  neighbourhood=NEIGHBOURHOOD,
                                  apply_rule=APPLY_RULE)


# Print the 2D matrices of every timestep
"""
index = 0
for i in cellular_automaton:
    print('Element',index)
    print(i)
    print('Size',i.shape)
    index +=1
"""
    
# Plot (at end of simulation)
cpl.plot2d(cellular_automaton)

# Plot (at particular timestep)
#cpl.plot2d(cellular_automaton, timestep=5)

# Plot (animation)
#cpl.plot2d_animate(cellular_automaton)

