from . import cellular_automata as ca
import numpy as np


GRIDSIZE_X = None
GRIDSIZE_Y = None
BETA = None
BETA_LEADER = None
P_OCCUPATION = None
P_OPINION_1 = None
S_LEADER = None 
H = None
S_MEAN = None
TIMESTEPS = None
S_LEADER_RANGE = None
EMP_RANGE = None
TEMP_RANGE = None

# Function for fig.3
# The function to pass to the pool
def simulate_single_run(t):
    model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=t, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN, show_tqdm=False)
    data = model.evolve(TIMESTEPS)
    simulation_final = data['opinions'][TIMESTEPS-1]
    a_T = model.mean_cluster_radius()
    return a_T

# Function for fig.4
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