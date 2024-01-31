"""
Do diagram 1 with values of medium grid
"""
import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt
import csv


# Step 0 - Find the critical temperature of the system
## Empirically (like shown in critical_temperature.py) or
## Analitically, using the approximations referenced in the CA paper
# Done in threshold_plot!!!
NUMBER_OF_SL_VALUES_TO_TEST = 10
SIMS_PER_SL_VALUE = 5


TIMESTEPS = 10

critical_temperature = 30

# Deterministic case params
NUMBER_OF_P1_VALUES_PER_SL = 1 # 3


params = []
reader = csv.DictReader(open('./figures/t_threshold_plot_params.csv'))
for row in reader:
    print(row)
    params.append(dict(row))
    # profit !

print(params)
params = params[0]


# Step 2 - Set system to the parameters used to find the critical temperature
GRIDSIZE_X,GRIDSIZE_Y = int(params['GRIDSIZE_X']),int(params['GRIDSIZE_Y'])
#TEMPERATURE = 0
BETA = float(params['BETA'])
BETA_LEADER = float(params['BETA_LEADER'])
H =  float(params['H'])
p =  float(params['P_OCCUPATION'])
p_1 = 0.5 # For the simulation at Tc, we have randomly innit grid!!!
INFLUENCE_LEADER = float(params['S_LEADER'])          # The one 
a_0 = float(params['a_0'])
INFLUENCE_DISTRIBUTION_MEAN = float(params['S_MEAN'])

################################

R = GRIDSIZE_X/2
S_L_min = ca.minimun_leader_strength(R,BETA,H)
S_L_max = ca.maximun_leader_strength(R,BETA,H)
cluster_min = ca.a(R,BETA,H,S_L_min)
cluster_max = ca.a(R,BETA,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5


# Iterate all SL values
SL_values = np.linspace(0,2*S_L_max,NUMBER_OF_SL_VALUES_TO_TEST)



# Deterministic case
print('Computing DETERMINISTIC POINTS (T=0)')


p_1_values = np.linspace(0,1,NUMBER_OF_P1_VALUES_PER_SL) # Different values of p_1 to run sims with
points_x_deter = np.zeros(len(p_1_values)*NUMBER_OF_SL_VALUES_TO_TEST)
points_y_deter = np.zeros(len(p_1_values)*NUMBER_OF_SL_VALUES_TO_TEST)

for index in range(NUMBER_OF_SL_VALUES_TO_TEST):

    S_LEADER = SL_values[index]
    print(f'SL value: {index+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    #Deterministic
    TEMP = 0
    
    for sim in range(len(p_1_values)):

        P_OPINION_1 = p_1_values[sim]

        model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta_leader=BETA_LEADER, beta=BETA, h=H, p_occupation=p, p_opinion_1=p_1, s_leader=S_LEADER, s_mean=INFLUENCE_DISTRIBUTION_MEAN)
        data = model.evolve(TIMESTEPS)
        #simulation = data['opinions']
        #cluster_sizes = data['cluster_sizes']
        last_cluster_size = data['cluster_sizes'][-1]

        # Save the point
        point_index = index*SIMS_PER_SL_VALUE + sim
        points_x_deter[point_index] = S_LEADER
        points_y_deter[point_index] = last_cluster_size

    # With temperature
    



# Create model out of it!