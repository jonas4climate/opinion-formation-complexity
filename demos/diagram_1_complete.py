"""
Replicates diagram 1 from paper
"""

import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

################################

NUMBER_OF_SL_VALUES_TO_TEST = 10
SIMS_PER_SL_VALUE = 10

NUMBER_OF_P1_VALUES_PER_SL = 3 # For the deterministic case


TEMPERATURE_VALUES_PER_SL = 3
TMAX = 100

# TODO: Add points with different innit, so we converge to other stable state
# TODO: Add sims with temperature so we can also get unstable states

################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 50
TEMP = 0
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 0
a_0 = 1 # Size of innitial cluster around leader
S_MEAN = 1

################################

R = GRIDSIZE_X/2
S_L_min = ca.minimun_leader_strength(R,BETA,H)
S_L_max = ca.maximun_leader_strength(R,BETA,H)
cluster_min = ca.a(R,BETA,H,S_L_min)
cluster_max = ca.a(R,BETA,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5

################################

#if TEMPERATURE == 0:
#    expect_cluster = ca.analytical_expect_clusters(
#        GRIDSIZE_X/2, BETA_PEOPLE, H, INFLUENCE_LEADER)
#    print('Expect clusters?', expect_cluster)

################################

# Iterate all SL values
SL_values = np.linspace(0,2*S_L_max,NUMBER_OF_SL_VALUES_TO_TEST)

################################

# 1. Deterministic case (T=0)
# We just need to try different innitializations: low p_1 and high p_1
# For each s_L value

"""
print('Computing DETERMINISTIC POINTS (T=0)')

p_1_values = np.linspace(0,1,NUMBER_OF_P1_VALUES_PER_SL)
TEMP = 0

points_x_deter = np.zeros(len(p_1_values)*NUMBER_OF_SL_VALUES_TO_TEST)
points_y_deter = np.zeros(len(p_1_values)*NUMBER_OF_SL_VALUES_TO_TEST)

for index in range(NUMBER_OF_SL_VALUES_TO_TEST):

    S_LEADER = SL_values[index]

    print(f'SL value: {index+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    for sim in range(len(p_1_values)):

        P_OPINION_1 = p_1_values[sim]

        model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta_leader=BETA_LEADER, beta=BETA, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
        data = model.evolve(TIMESTEPS)
        #simulation = data['opinions']
        #cluster_sizes = data['cluster_sizes']
        last_cluster_size = data['cluster_sizes'][-1]

        # Save the point
        point_index = index*len(p_1_values) + sim
        points_x_deter[point_index] = S_LEADER
        points_y_deter[point_index] = last_cluster_size
"""


################################

# 2. Stochastic case (T>0)
# With same innitializations: low p_1 and high p_1
# We will try different values of T

print('Computing STOCHSTIC POINTS (T>0)')

p_1_values = np.linspace(0,1,NUMBER_OF_P1_VALUES_PER_SL)
temperatures = np.linspace(0,TMAX,TEMPERATURE_VALUES_PER_SL) # Ignore the first value, T=0
TEMP = 0

#points_x_stoc = np.zeros(len(temperatures)*len(p_1_values)*NUMBER_OF_SL_VALUES_TO_TEST)
#points_y_stoc = np.zeros(len(temperatures)*len(p_1_values)*NUMBER_OF_SL_VALUES_TO_TEST)

"""
for index in range(NUMBER_OF_SL_VALUES_TO_TEST):

    S_LEADER = SL_values[index]

    print(f'SL value: {index+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    # Test all temperatures
    for index2 in range(NUMBER_OF_TEMPERATURE_VALUES-1):

        TEMP = temperatures[index2]
        # Test all p1 values!
        for sim in range(len(p_1_values)):

            P_OPINION_1 = p_1_values[sim]

            model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta_leader=BETA_LEADER, beta=BETA, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
            data = model.evolve(TIMESTEPS)
            #simulation = data['opinions']
            #cluster_sizes = data['cluster_sizes']
            last_cluster_size = data['cluster_sizes'][-1]

            # Save the point
            point_index = index*NUMBER_OF_SL_VALUES_TO_TEST + index2*NUMBER_OF_TEMPERATURE_VALUES + sim
            points_x_stoc[point_index] = S_LEADER
            points_y_stoc[point_index] = last_cluster_size
"""

# x,y coordinates of all of them
points_x_stoc = np.zeros(NUMBER_OF_SL_VALUES_TO_TEST*TEMPERATURE_VALUES_PER_SL*SIMS_PER_SL_VALUE)
points_y_stoc = np.zeros(NUMBER_OF_SL_VALUES_TO_TEST*TEMPERATURE_VALUES_PER_SL*SIMS_PER_SL_VALUE)


# Do stoch sim points for all SL
for i in range(NUMBER_OF_SL_VALUES_TO_TEST):
    S_LEADER = SL_values[i]
    print(f'SL value: {i+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    # For all temperatures, which includes t=0
    for j in range(TEMPERATURE_VALUES_PER_SL):
        TEMP = temperatures[j]

        # Many times per temperature
        for k in range(SIMS_PER_SL_VALUE):
            model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta_leader=BETA_LEADER, beta=BETA, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
            data = model.evolve(TIMESTEPS)
            #simulation = data['opinions']
            #cluster_sizes = data['cluster_sizes']
            last_cluster_size = data['cluster_sizes'][-1]
            
            # If not, save thereç
            index = i*NUMBER_OF_SL_VALUES_TO_TEST + j*TEMPERATURE_VALUES_PER_SL + k*SIMS_PER_SL_VALUE
            points_x_stoc[index] = S_LEADER
            points_y_stoc[index] = last_cluster_size



"""
for index in range(NUMBER_OF_SL_VALUES_TO_TEST):
    S_LEADER = SL_values[index]
    print(f'SL value: {index+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    # Test all temperatures
    for index2 in range(NUMBER_OF_TEMPERATURE_VALUES-1):
        TEMP = temperatures[index2]

        # Test all p1 values!
        for sim in range(len(p_1_values)):
            P_OPINION_1 = p_1_values[sim]

            model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta_leader=BETA_LEADER, beta=BETA, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
            data = model.evolve(TIMESTEPS)
            last_cluster_size = data['cluster_sizes'][-1]

            # Save the point
            point_index = index*(NUMBER_OF_SL_VALUES_TO_TEST-1) + index2*(NUMBER_OF_TEMPERATURE_VALUES-1) + sim
            points_x_stoc[point_index] = S_LEADER
            points_y_stoc[point_index] = last_cluster_size
"""

################################



"""


# Get matrices for answers
points_x = np.zeros(NUMBER_OF_SL_VALUES_TO_TEST*SIMS_PER_SL_VALUE)
points_y = np.zeros(NUMBER_OF_SL_VALUES_TO_TEST*SIMS_PER_SL_VALUE)



for index in range(NUMBER_OF_SL_VALUES_TO_TEST):

    INFLUENCE_LEADER = SL_values[index]

    print(f'Sim {index+1}/{NUMBER_OF_SL_VALUES_TO_TEST}')

    for sim in range(SIMS_PER_SL_VALUE):

        model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
        data = model.evolve(TIMESTEPS)
        #simulation = data['opinions']
        #cluster_sizes = data['cluster_sizes']
        last_cluster_size = data['cluster_sizes'][-1]

        # Save the point
        point_index = index*SIMS_PER_SL_VALUE + sim
        points_x[point_index] = INFLUENCE_LEADER
        points_y[point_index] = last_cluster_size

"""
################################

# Plot diagram
fig, ax = ca.plot_diagram(R,BETA,H)

# 'o'   
#ax.scatter(points_x_deter, points_y_deter,label='T=0', facecolors='none', edgecolors='green')
ax.scatter(points_x_stoc, points_y_stoc,label='T>0', facecolors='none', edgecolors='red')


# Add points
ax.set_xlim(0,2*S_L_max)
ax.set_ylim(0,int(R)+1)

#ax.scatter(points_x, points_y)

# Show
plt.grid()
plt.legend()
plt.tight_layout()
plt.show(block=True)


