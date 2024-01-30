""""
Plot fig.3 of the paper
Used the equation (9) and (10), assuming that the mean cluster radius a_T calculated by the proportion of number of cells present 1, which means:
a_T_proportion = (number of cells have 1) / area of the circle with radius R(R=GRIDSIZE_X/2)

Added a new function for calculate the mean cluster radius

Caution: Just focus on the gird of final timestep 

TODO: Still need to be tested for the optimum parameter
TODO: Speed up the simulation (if can)
"""

import numpy as np
import ca.CA_module as ca
import matplotlib.pyplot as plt
from numpy import *

# Parameters
GRIDSIZE_X,GRIDSIZE_Y = 45,45
TIMESTEPS = 10
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 0

INFLUENCE_LEADER = 400   
INFLUENCE_DISTRIBUTION_MEAN = 1


TEMPERATURE=np.linspace(0,40,10)
simulation_times = 10    # Run times
Mean_cluster_radius = []


for t in TEMPERATURE:
    Simu_mean_cluster_radius = []
    
    for s in range(simulation_times):
        model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, t, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
        data = model.evolve(TIMESTEPS)
        simulation_final = data['opinions'][TIMESTEPS-1]
        #print("final",simulation_final)
        
        #########plot the grid of final timestep
        # fig, ax = plt.subplots()
        # im = ax.imshow(simulation_final, cmap='seismic',
        #            interpolation='nearest', vmin=-1, vmax=1)
    
        
        # plt.tight_layout()
        # plt.show(block=False)
        # plt.pause(0.2)
        # ax.clear()
        #########
        
        a_T = model.mean_cluster_radius()
        Simu_mean_cluster_radius.append(a_T)
    
    Mean_cluster_radius.append(mean(Simu_mean_cluster_radius))      

print("Mean_cluster_radius",Mean_cluster_radius)
    

# Plotting
plt.suptitle(' Mean cluster radius a vs. temperature T')
plt.title(f'S_L={INFLUENCE_LEADER},simulation={simulation_times},GRIDSIZE={GRIDSIZE_X},H={H}')
plt.xlabel('T')
plt.ylabel('a(T)')

xmin,xmax = 0,40
ymin,ymax = 0,10
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.plot(TEMPERATURE,Mean_cluster_radius,marker="o")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()