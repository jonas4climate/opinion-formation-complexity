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
import demos.ca.cellular_automata as ca
import matplotlib.pyplot as plt

# Parameters
GRIDSIZE_X,GRIDSIZE_Y = 45,45
TIMESTEPS = 10
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 0

S_LEADER = 400   
S_MEAN = 1


TEMPERATURE=np.linspace(0,40,10)
simulation_times = 8    # Run times
Mean_cluster_radius = []


for t in TEMPERATURE:
    Simu_mean_cluster_radius = []
    
    for s in range(simulation_times):
        model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=t, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
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
    
    Mean_cluster_radius.append(np.mean(Simu_mean_cluster_radius))      

print("Mean_cluster_radius",Mean_cluster_radius)
    

# Plotting
plt.suptitle(' Mean cluster radius a vs. temperature T')
plt.title(f'S_L={S_LEADER}')
plt.xlabel('T')
plt.ylabel('a(T)')

# xmin,xmax = 0,30
# ymin,ymax = 0,10
# plt.xlim([xmin,xmax])
# plt.ylim([ymin,ymax])
plt.plot(TEMPERATURE,Mean_cluster_radius,c='black',linestyle='--')

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()