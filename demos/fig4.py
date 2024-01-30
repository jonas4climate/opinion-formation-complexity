
""""
Plot fig.4 of the paper

TODO: Still need to be tested for the optimum parameter
TODO: Speed up the simulation (if can)
"""
import numpy as np
import matplotlib.pyplot as plt
import ca.cellular_automata as ca

# Parameters
GRIDSIZE_X, GRIDSIZE_Y = 45, 45
TIMESTEPS = 10
BETA = 1
BETA_LEADER = 1
H = 0

P_OCCUPATION = 1
P_OPINION_1 = 0
S_MEAN = 1

NUM_OF_S_LEADER_RANGE = 7
NUM_OF_TEMP_RANGE = 10
MAX_S_LEADER = 500
MAX_TEMP = 120


S_LEADER_RANGE = np.linspace(0, MAX_S_LEADER, NUM_OF_S_LEADER_RANGE) 
TEMP_RANGE = np.linspace(0, MAX_TEMP, NUM_OF_TEMP_RANGE) 

SIMULATION_TIMES = 5   # Run times


all_critical_temperatures = [[0] * NUM_OF_S_LEADER_RANGE for _ in range(SIMULATION_TIMES)]  #2D list for calculate the final average cirtical temperature 

#print("all_critical_temperatue",all_critical_temperatures)

final_critical_temperatures = []

for s in range(SIMULATION_TIMES):

    critical_temperatures = []
        
    for influence_leader in S_LEADER_RANGE:
        
        order_parameter = []
        
        for temperature in TEMP_RANGE:
            
            model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=temperature,
                          beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION,
                          p_opinion_1=P_OPINION_1, s_leader=influence_leader, s_mean=S_MEAN)

            data = model.evolve(TIMESTEPS)
            
            
            final_opinions = data['opinions'][-1]
            order_param = np.mean(final_opinions == 1) 
            order_parameter.append(order_param)

        #Find Critical temperature
        d_order_param = np.gradient(order_parameter, TEMP_RANGE)
        critical_temperature_idx = np.argmax(np.abs(d_order_param))
        critical_temperature = TEMP_RANGE[critical_temperature_idx]
        
        critical_temperatures.append(critical_temperature)
        
    all_critical_temperatures[s] = critical_temperatures


print("all_critical_temperatue",all_critical_temperatures)



# Plotting
plt.plot(S_LEADER_RANGE, final_critical_temperatures, marker='o')
plt.xlabel('Leader Influence Strength')
plt.ylabel('Critical Temperature')
plt.suptitle('Critical Temperature vs Leader Influence Strength')
plt.title(f'GRIDSIZE={GRIDSIZE_X},simulation={SIMULATION_TIMES},H={H}')
plt.grid(True)
plt.show()
