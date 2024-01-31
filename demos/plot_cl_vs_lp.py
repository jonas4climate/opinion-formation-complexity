from network.network import Network
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
Runs the network for a specified number of runs with varying c_L parameters
"""



#Use same seed for initialization
np.random.seed(0)


#Set parameters
################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 20
TEMP = 10
H = 0
BETA = 1
BETA_LEADER = 1
P_OCCUPATION = 1
P_OPINION_1 = 1

#Grid
S_LEADER = 300   # Leader influence
S_MEAN = 1

NETWORK_TYPE = 'barabasi-albert'

################################

#Make sure to store parameters
parameters = {
    'GRIDSIZE_X': GRIDSIZE_X,
    'GRIDSIZE_Y': GRIDSIZE_Y,
    'TIMESTEPS': TIMESTEPS,
    'TEMP': TEMP,
    'H': H,
    'BETA': BETA,
    'BETA_LEADER': BETA_LEADER,
    'P_OCCUPATION': P_OCCUPATION,
    'P_OPINION_1': P_OPINION_1,
    'S_LEADER': S_LEADER,
    'S_MEAN': S_MEAN,
}

#Define C_L range

C_L_LOWER = 1
C_L_UPPER = 50
C_L_RANGE = range(C_L_LOWER,C_L_UPPER+1)

NR_RUNS = 5

# Store values in array
average_path_lengths = []
std_dev_path_lengths = []

for C in C_L_RANGE:
    path_lengths_per_run = []

    for run in range(NR_RUNS):
        network = Network(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER,
                          h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN,
                          network_type=NETWORK_TYPE, c_leader=C)
        node_count = nx.number_of_nodes(network.G)

        data = network.evolve(TIMESTEPS)
        longest_path_length = len(data['longest_path'][-1])
        path_lengths_per_run.append(longest_path_length)

    # Calculate the average of all accumulated path lengths and the standard deviation
    average_length = np.mean(path_lengths_per_run)
    std_dev_length = np.std(path_lengths_per_run)

    average_path_lengths.append(average_length)
    std_dev_path_lengths.append(std_dev_length)

# Plotting
plt.plot(C_L_RANGE, average_path_lengths, label='Average Path Length')
plt.fill_between(C_L_RANGE, np.subtract(average_path_lengths, std_dev_path_lengths), np.add(average_path_lengths, std_dev_path_lengths), alpha=0.1, label='Standard Deviation', color='blue')

# Adding labels and title
plt.suptitle('Effect of leader connectivity on average longest path length of graph')
plt.title(f'N={node_count}, {NR_RUNS} runs/$C_l$, $C_l$_MAX={C_L_UPPER}')

plt.xlabel('$C_l$')
plt.ylabel('Average Longest Path Length')


# Display the plot
plt.show()



