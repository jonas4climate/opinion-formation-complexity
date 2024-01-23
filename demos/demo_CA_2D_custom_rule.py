"""
Basic code to create a custom 2D CA rule 
based on documentation: https://cellpylib.org/additional.html#custom-rules

Should be modified to implement the paper rule!

According to https://cellpylib.org/colors.html the number of states, or colors, that a cell can adopt is given by k. 
For example, a binary cellular automaton, in which a cell can assume only values of 0 and 1, has k = 2. Also,
the rule number is given in base 10 but is interpreted as the rule in base k (thus rule 777 corresponds to ‘1001210’ when k = 3).







"""
import cellpylib as cpl
from collections import defaultdict


class CustomRule(cpl.BaseRule):

    def __init__(self):
        self.count = defaultdict(int)

    def __call__(self, n, c, t):
        self.count[c] += 1
        return self.count[c]



# Global parameters
GRIDSIZE_X,GRIDSIZE_Y = 30,30
TIMESTEPS = 5
NEIGHBOURHOOD = 'Moore'
APPLY_RULE = CustomRule()
#APPLY_RULE = lambda n, c,t: cpl.totalistic_rule(n, k=2, rule=126) # This is the core of what we need to modify to match the paper
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
#cpl.plot2d(cellular_automaton)

# Plot (at particular timestep)
#cpl.plot2d(cellular_automaton, timestep=5)

# Plot (animation)
cpl.plot2d_animate(cellular_automaton)

