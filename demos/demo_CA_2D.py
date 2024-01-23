"""
Basic CA code to ensure 2D CA can be run by everybody
Taken from documentation: https://cellpylib.org/twodim.html
"""

import cellpylib as cpl

# initialize a 60x60 2D cellular automaton
cellular_automaton = cpl.init_simple2d(60, 60)

# evolve the cellular automaton for 30 time steps,
#  applying totalistic rule 126 to each cell with a Moore neighbourhood
cellular_automaton = cpl.evolve2d(cellular_automaton, timesteps=30, neighbourhood='Moore',
                                  apply_rule=lambda n, c, t: cpl.totalistic_rule(n, k=2, rule=126))


# Normal plot
#cpl.plot2d(cellular_automaton)

# Plot at particular timestep
# cpl.plot2d(cellular_automaton, timestep=10)


# Plot animation
cpl.plot2d_animate(cellular_automaton)
