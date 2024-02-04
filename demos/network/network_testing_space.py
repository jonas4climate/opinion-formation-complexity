from network import Network
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv

np.random.seed(0)
################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 30
TEMP = 10
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 1

#Grid
S_LEADER = 150  # Leader influence
S_MEAN = 1

#Barabasi-Albert
C_LEADER = 20

NETWORK_TYPE = 'grid'
network = Network(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN , network_type = NETWORK_TYPE, c_leader = C_LEADER)
data = network.evolve(TIMESTEPS)
network.plot_opinion_network_evolution(data, interval=250)
plt.show()
# First we model the network as a grid with both T=0, T=25

NETWORK_TYPE = 'grid'
network = Network(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN , network_type = NETWORK_TYPE, c_leader = C_LEADER)
data = network.evolve(TIMESTEPS)
network.plot_opinion_network_evolution(data, interval=10, save = True)

NETWORK_TYPE = 'barabasi-albert'

# Identify used parameters
params_used = {
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

#Define C_L range, number of runs

C_L_LOWER = 90
C_L_UPPER = 100
C_L_RANGE = range(C_L_LOWER,C_L_UPPER+1)

NR_RUNS = 10

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
plt.title(f'N={node_count}, {NR_RUNS} runs/$C_l$, $S_l$={S_LEADER}, $\\hat{{S}}$={S_MEAN}')

plt.xlabel('$C_l$')
plt.ylabel('Average Longest Path Length')

title_string = f'Cl_{C_L_RANGE}_{NR_RUNS}_runs'
plt.savefig(title_string, dpi=300, bbox_inches='tight')

with open('./figures/BA_network_params.csv', 'w') as f:
    w = csv.DictWriter(f, params_used.keys())
    w.writeheader()
    w.writerow(params_used)
