
""""
Plot fig.4 of the paper

TODO: Still need to be tested for the optimum parameter
TODO: Speed up the simulation (if can)
"""
import numpy as np
import matplotlib.pyplot as plt
import cellular_automata as ca
from multiprocessing import Pool
from tqdm import tqdm
import csv
# Parameters
GRIDSIZE_X, GRIDSIZE_Y = 45 , 45
TIMESTEPS = 10
BETA = 1
BETA_LEADER = 1
H = 0

P_OCCUPATION = 1
P_OPINION_1 = 0
S_MEAN = 1

NUM_OF_S_LEADER_RANGE = 8
NUM_OF_TEMP_RANGE = 10
MAX_S_LEADER = 500
MAX_TEMP = 120


S_LEADER_RANGE = np.linspace(0, MAX_S_LEADER, NUM_OF_S_LEADER_RANGE) 
TEMP_RANGE = np.linspace(0, MAX_TEMP, NUM_OF_TEMP_RANGE) 

SIMULATION_TIMES = 10   # Run times



def simulate(_):
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

        d_order_param = np.gradient(order_parameter, TEMP_RANGE)
        critical_temperature_idx = np.argmax(np.abs(d_order_param))
        critical_temperatures.append(TEMP_RANGE[critical_temperature_idx])

    return critical_temperatures

if __name__ == "__main__":
    with Pool() as pool:
        all_critical_temperatures = pool.map(simulate, range(SIMULATION_TIMES))

    mean_critical_temps = np.mean(all_critical_temperatures, axis=0)
    std_critical_temps = np.std(all_critical_temperatures, axis=0)
    
    # Write parameters and results to CSV
    with open('data/fig4_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['GRIDSIZE_X', GRIDSIZE_X])
        writer.writerow(['GRIDSIZE_Y', GRIDSIZE_Y])
        writer.writerow(['TIMESTEPS', TIMESTEPS])
        writer.writerow(['BETA', BETA])
        writer.writerow(['BETA_LEADER', BETA_LEADER])
        writer.writerow(['H', H])
        writer.writerow(['P_OCCUPATION', P_OCCUPATION])
        writer.writerow(['P_OPINION_1', P_OPINION_1])
        writer.writerow(['S_MEAN', S_MEAN])
        writer.writerow(['NUM_OF_S_LEADER_RANGE', NUM_OF_S_LEADER_RANGE])
        writer.writerow(['NUM_OF_TEMP_RANGE', NUM_OF_TEMP_RANGE])
        writer.writerow(['MAX_S_LEADER', MAX_S_LEADER])
        writer.writerow(['MAX_TEMP', MAX_TEMP])
        writer.writerow(['SIMULATION_TIMES', SIMULATION_TIMES])
        writer.writerow([])
        writer.writerow(['Leader Influence Strength', 'Mean Critical Temperature', 'Standard Deviation'])
        for i, s_leader in enumerate(S_LEADER_RANGE):
            writer.writerow([s_leader, mean_critical_temps[i], std_critical_temps[i]])

    

    # Plotting
    plt.plot(S_LEADER_RANGE, mean_critical_temps, marker='o')
    plt.fill_between(S_LEADER_RANGE, np.array(mean_critical_temps)-np.array(std_critical_temps), np.array(mean_critical_temps)+np.array(std_critical_temps), alpha=0.3)
    plt.xlabel('Leader Influence Strength')
    plt.ylabel('Critical Temperature')
    plt.suptitle('Critical Temperature vs Leader Influence Strength')
    plt.title(f'GRIDSIZE={GRIDSIZE_X}, simulation={SIMULATION_TIMES}, H={H}')
    plt.grid(True)
    plt.show()
    

######## formal function
# all_critical_temperatures = [[0] * NUM_OF_S_LEADER_RANGE for _ in range(SIMULATION_TIMES)]  #2D list for calculate the final average cirtical temperature 

# print("all_critical_temperatue",all_critical_temperatures)

# final_critical_temperatures = []

# for s in range(SIMULATION_TIMES):

#     critical_temperatures = []
        
#     for influence_leader in S_LEADER_RANGE:
        
#         order_parameter = []
        
#         for temperature in TEMP_RANGE:
            
#             model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=temperature,
#                           beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION,
#                           p_opinion_1=P_OPINION_1, s_leader=influence_leader, s_mean=S_MEAN)

#             data = model.evolve(TIMESTEPS)
            
            
#             final_opinions = data['opinions'][-1]
#             order_param = np.mean(final_opinions == 1) 
#             order_parameter.append(order_param)

#         #Find Critical temperature
#         d_order_param = np.gradient(order_parameter, TEMP_RANGE)
#         critical_temperature_idx = np.argmax(np.abs(d_order_param))
#         critical_temperature = TEMP_RANGE[critical_temperature_idx]
        
#         critical_temperatures.append(critical_temperature)
        
#     all_critical_temperatures[s] = critical_temperatures

# print("all_critical_temperatue",all_critical_temperatures)

# for n in range(NUM_OF_S_LEADER_RANGE):
#     temperature_collect = []
#     for s in range(SIMULATION_TIMES):
#         temperature_collect.append(all_critical_temperatures[s][n])

#     final_critical_temperatures.append(np.mean(temperature_collect))

# print("final_cri_tem", final_critical_temperatures)


# # Plotting
# plt.plot(S_LEADER_RANGE, final_critical_temperatures, marker='o')
# plt.xlabel('Leader Influence Strength')
# plt.ylabel('Critical Temperature')
# plt.suptitle('Critical Temperature vs Leader Influence Strength')
# plt.title(f'GRIDSIZE={GRIDSIZE_X},simulation={SIMULATION_TIMES},H={H}')
# plt.grid(True)
# plt.show()

#####################