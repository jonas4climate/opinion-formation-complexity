
""""
Plot fig.4 of the paper

TODO: Still need to be tested for the optimum parameter
TODO: Speed up the simulation (if can)
"""
import numpy as np
import matplotlib.pyplot as plt
import ca.CA_module as ca

# Parameters
GRIDSIZE_X, GRIDSIZE_Y = 55, 55
TIMESTEPS = 10
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 0


INFLUENCE_DISTRIBUTION_MEAN = 1

INFLUENCE_LEADER_RANGE = np.linspace(0, 500, 8) 
TEMPERATURE_RANGE = np.linspace(0, 120, 10) 


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

# Plotting
plt.plot(INFLUENCE_LEADER_RANGE, critical_temperatures, marker='o')
plt.xlabel('Leader Influence Strength')
plt.ylabel('Critical Temperature')
plt.title('Critical Temperature vs Leader Influence Strength')
plt.grid(True)
plt.show()
