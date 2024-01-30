
""""
Plot fig.4 of the paper

TODO: Still need to be tested for the optimum parameter
TODO: Speed up the simulation (if can)
"""
import numpy as np
import matplotlib.pyplot as plt
import ca.CA_module as ca

# Parameters
GRIDSIZE_X, GRIDSIZE_Y = 45, 45
TIMESTEPS = 10
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 0
INFLUENCE_DISTRIBUTION_MEAN = 1

NUM_OF_INFLUENCE_LEADER_RANGE = 7
NUM_OF_TEMPERATURE_RANGE = 10
MAX_LEADER_INFLUNCE = 500
MAX_TEMPERATURE = 120

INFLUENCE_LEADER_RANGE = np.linspace(0, MAX_LEADER_INFLUNCE, NUM_OF_INFLUENCE_LEADER_RANGE) 
TEMPERATURE_RANGE = np.linspace(0, MAX_TEMPERATURE, NUM_OF_TEMPERATURE_RANGE) 

SIMULATION_TIMES = 5   # Run times

all_critical_temperatures = [[0] * NUM_OF_INFLUENCE_LEADER_RANGE for _ in range(SIMULATION_TIMES)]  #2D list for calculate the final average cirtical temperature 

#print("all_critical_temperatue",all_critical_temperatures)

final_critical_temperatures = []

for s in range(SIMULATION_TIMES):

    critical_temperatures = []
        
    for influence_leader in INFLUENCE_LEADER_RANGE:
        
        order_parameter = []
        
        for temperature in TEMPERATURE_RANGE:
            
            model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, temperature, BETA_LEADER, BETA_PEOPLE, H,
                    p, p_1, influence_leader, INFLUENCE_DISTRIBUTION_MEAN,
                    ca.euclidean_distance, ca.prob_dist_influence_people)

            data = model.evolve(TIMESTEPS)
            
            
            final_opinions = data['opinions'][-1]
            order_param = np.mean(final_opinions == 1) 
            order_parameter.append(order_param)

        #Find Critical temperature
        d_order_param = np.gradient(order_parameter, TEMPERATURE_RANGE)
        critical_temperature_idx = np.argmax(np.abs(d_order_param))
        critical_temperature = TEMPERATURE_RANGE[critical_temperature_idx]
        
        critical_temperatures.append(critical_temperature)
        
    all_critical_temperatures[s] = critical_temperatures


print("all_critical_temperatue",all_critical_temperatures)

for n in range(NUM_OF_INFLUENCE_LEADER_RANGE):
    temperature_collect = []
    for s in range(SIMULATION_TIMES):
        temperature_collect.append(all_critical_temperatures[s][n])
    
    final_critical_temperatures.append(np.mean(temperature_collect))
    
print("final_cri_tem", final_critical_temperatures)

# Plotting
plt.plot(INFLUENCE_LEADER_RANGE, final_critical_temperatures, marker='o')
plt.xlabel('Leader Influence Strength')
plt.ylabel('Critical Temperature')
plt.suptitle('Critical Temperature vs Leader Influence Strength')
plt.title(f'GRIDSIZE={GRIDSIZE_X},simulation={SIMULATION_TIMES},H={H}')
plt.grid(True)
plt.show()
