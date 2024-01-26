"""

Creates a threshold plot in terms of temperature:
probability of overcoming leader's opinion as temperature increases

note that for this scenario, everybody has leaders opinion!

p_overcoming_leader = f(TEMPERATURE)

"""

import CA_module as ca
import numpy as np
import matplotlib.pyplot as plt

################################

NUMBER_OF_T_VALUES_TO_TEST = 20
SIMS_PER_T_VALUE = 10
T_MAX = 100

THRESHOLD = 3 # Maximun leader cluster radius that is not considered opinion overcomming

################################

# Todo this we first get parameters that for T=0
# ensure either unification with leader

################################

GRIDSIZE_X,GRIDSIZE_Y = 15,15
TIMESTEPS = 5
#TEMPERATURE = 0
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 1 # In this scenario everybody believes the leader at start
a_0 = 1
INFLUENCE_LEADER = 50             # May need to tweak this!
INFLUENCE_DISTRIBUTION_MEAN = 1

################################

R = GRIDSIZE_X/2
S_L_min = ca.minimun_leader_strength(R,BETA_PEOPLE,H)
S_L_max = ca.maximun_leader_strength(R,BETA_PEOPLE,H)
cluster_min = ca.a(R,BETA_PEOPLE,H,S_L_min)
cluster_max = ca.a(R,BETA_PEOPLE,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5

################################


# Do one simulation test first to ensure that
# to start with we actually have a cluster to overcome
TEMPERATURE = 0

model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
data = model.evolve(TIMESTEPS)
simulation = data['opinions']
cluster_sizes = data['cluster_sizes']
last_cluster = cluster_sizes[-1]

# Do diagram plot to ensure we are in the right region
fig, ax = ca.plot_diagram(R,BETA_PEOPLE,H)

ax.set_xlim([0,2*S_L_max])
ax.set_ylim([0,int(R)+1])

# Plot last cluster
ax.scatter(INFLUENCE_LEADER,last_cluster)

# Show
plt.grid()
plt.tight_layout()
plt.show(block=True)

################################

fig, ax = plt.subplots()

grid_t = simulation[-1,:,:]
im = ax.imshow(grid_t, cmap='seismic',
                interpolation='nearest', vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

################################

# Do many simulations with many temperatures


temperatures = np.linspace(0,T_MAX,NUMBER_OF_T_VALUES_TO_TEST)
p_overcoming_leader = np.zeros(NUMBER_OF_T_VALUES_TO_TEST)

for index in range(NUMBER_OF_T_VALUES_TO_TEST):

    TEMPERATURE = temperatures[index]
    print(f'Sim {index+1}/{NUMBER_OF_T_VALUES_TO_TEST}')
    leader_overcomed = 0

    for sim in range(SIMS_PER_T_VALUE):
        model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
        data = model.evolve(TIMESTEPS)
        #simulation = data['opinions']
        #cluster_sizes = data['cluster_sizes']
        last_cluster_size = data['cluster_sizes'][-1]

        # Check if unification was overcomed!
        # That is, if the cluster around leader is gone
        # Which will be under a threshold
        if last_cluster_size <= THRESHOLD:
            leader_overcomed += 1
        

    # Compute probability of overcoming leader
    # Which is the number of sims that ended in unification
    # Over SIMS_PER_T_VALUE
    p_overcoming_leader[index] = leader_overcomed / SIMS_PER_T_VALUE

################################

# Plot the threshold phenomena

plt.figure()
plt.plot(temperatures,p_overcoming_leader)

plt.xlim([0,T_MAX])
plt.ylim([0,1])

plt.title('Effect of temperature on overcoming leader consensus')
plt.xlabel('Temperature')
plt.ylabel('p(Overcoming leader)')


plt.show()