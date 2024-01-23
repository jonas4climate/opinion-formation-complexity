"""
Basic CA code to ensure 2D CA can be run by everybody
Based on the documentation: https://cellpylib.org/twodim.html
"""

import cellpylib as cpl


# Global parameters
GRIDSIZE_X,GRIDSIZE_Y = 30,30
TIMESTEPS = 10
NEIGHBOURHOOD = 'Moore'

 # This is the core of what we need to modify to match the paper
APPLY_RULE = lambda n, c,t: cpl.totalistic_rule(n, k=2, rule=126)

# This should be a numpy.ndarray, and should also be modified to match the paper
STARTING_GRID = cpl.init_simple2d(GRIDSIZE_X, GRIDSIZE_Y) 


# Evolve the CA for the TIMESTEPS
# This returns a np.array that has TIMESTEPS numpyarrays of size (GRIDSIZE_X,GRIDSIZE_Y)
# With each grid element having one of k possible values
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


# Basic plots
#cpl.plot2d(cellular_automaton) # At end of simulation
#cpl.plot2d(cellular_automaton, timestep=5) # At a particular timestep
cpl.plot2d_animate(cellular_automaton) # Animation

# Plot code is available here, so we can modify it to fit our needs:
# https://github.com/lantunes/cellpylib/blob/master/cellpylib/ca_functions2d.py


# Custom rules