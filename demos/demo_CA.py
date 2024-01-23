"""
Basic CA code to ensure environment and package installation works correctly for everybody
Minimum example from documentation: https://cellpylib.org/working.html
"""

import cellpylib as cpl

cellular_automaton = cpl.init_simple(200)

cellular_automaton = cpl.evolve(cellular_automaton, timesteps=100, memoize=True,
                                apply_rule=lambda n, c, t: cpl.nks_rule(n, 30))

cpl.plot(cellular_automaton)

