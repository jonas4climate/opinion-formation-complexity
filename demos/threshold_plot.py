"""

Creates a threshold plot in terms of temperature:
probability of overcoming leader's opinion as temperature increases

note that for this scenario, everybody has leaders opinion!

p_overcoming_leader = f(TEMPERATURE)

"""

import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt
import csv

################################

NUMBER_OF_T_VALUES_TO_TEST = 10
SIMS_PER_T_VALUE = 20
T_MAX = 100

THRESHOLD = 5 # Maximun leader cluster radius that is not considered opinion overcomming
S_LEADER = 100             # May need to tweak this to ensure we are on cluster region!
################################

# Todo this we first get parameters that for T=0
# ensure either unification with leader

################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 20
#TEMPERATURE = 0
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 1 # In this scenario everybody believes the leader at start
a_0 = 1

S_MEAN = 1



R = GRIDSIZE_X/2
S_L_min = ca.minimun_leader_strength(R,BETA,H)
S_L_max = ca.maximun_leader_strength(R,BETA,H)
cluster_min = ca.a(R,BETA,H,S_L_min)
cluster_max = ca.a(R,BETA,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5

################################

# Save parameters

params_used = {'NUMBER_OF_T_VALUES_TO_TEST':NUMBER_OF_T_VALUES_TO_TEST,
               'SIMS_PER_T_VALUE':SIMS_PER_T_VALUE,
               'T_MAX':T_MAX,
               'THRESHOLD':THRESHOLD,
               'S_LEADER':S_LEADER,
               'GRIDSIZE_X': 21,
               'GRIDSIZE_Y':21,
               'TIMESTEPS':TIMESTEPS,
               'BETA':BETA,
               'BETA_LEADER':BETA_LEADER,
               'H':H,
               'P_OCCUPATION':P_OCCUPATION,
               'P_OPINION_1':P_OPINION_1,
               'a_0':a_0,
               'S_MEAN':S_MEAN,
               'R':R,
               'S_L_min':S_L_min,
               'S_L_max':S_L_max,
               'cluster_min':cluster_min,
               'cluster_max':cluster_max,
               'xmin':xmin,
               'xmax':xmax,
               'ymin':ymin,
               'ymax':ymax
               }


with open('./figures/t_threshold_plot_params.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, params_used.keys())
    w.writeheader()
    w.writerow(params_used)

# To read
#reader = csv.DictReader(open('myfile.csv'))
#for row in reader:
#    # profit !

################################


# Do one simulation test first to ensure that
# to start with we actually have a cluster to overcome
TEMP = 0

model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
data = model.evolve(TIMESTEPS)
simulation = data['opinions']
cluster_sizes = data['cluster_sizes']
last_cluster = cluster_sizes[-1]

# Do diagram plot to ensure we are in the right region
fig, ax = ca.plot_diagram(R,BETA,H)

ax.set_xlim(0,2*S_L_max)
ax.set_ylim(0,int(R)+1)

# Plot last cluster
ax.scatter(S_LEADER,last_cluster)

# Show
plt.grid()
plt.tight_layout()

plt.savefig('./figures/t_threshold_plot_start_analyticall.png')

plt.show(block=True)

################################

fig, ax = plt.subplots()

grid_t = simulation[-1,:,:]
im = ax.imshow(grid_t, cmap='seismic',
                interpolation='nearest', vmin=-1, vmax=1)
plt.tight_layout()

plt.savefig('./figures/t_threshold_plot_start_grid.png')

plt.show()

################################

# Do many simulations with many temperatures


temperatures = np.linspace(0,T_MAX,NUMBER_OF_T_VALUES_TO_TEST)
p_overcoming_leader = np.zeros(NUMBER_OF_T_VALUES_TO_TEST)

for index in range(NUMBER_OF_T_VALUES_TO_TEST):

    TEMP = temperatures[index]
    print(f'Sim {index+1}/{NUMBER_OF_T_VALUES_TO_TEST}')
    leader_overcomed = 0

    for sim in range(SIMS_PER_T_VALUE):
        model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
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

print(p_overcoming_leader)


# Save the data
np.savetxt('./figures/t_threshold_plot_temperatures.npy',temperatures,delimiter=",")
np.savetxt('./figures/t_threshold_plot_tp_overcoming_leader.npy',p_overcoming_leader,delimiter=",")

# Plot the threshold phenomena

plt.figure()
plt.plot(temperatures,p_overcoming_leader,lw=2,c='blue')


plt.xlim([0,T_MAX])
plt.ylim([0,1])

plt.suptitle('Effect of temperature on overcoming leader consensus')
plt.title(f'R={R}, {SIMS_PER_T_VALUE} runs/T, T_MAX={T_MAX},THRESHOLD={THRESHOLD}')
plt.xlabel('Temperature')
plt.ylabel('p(Overcoming leader)')

plt.grid()
plt.tight_layout()


plt.savefig('./figures/t_threshold_plot.png')


plt.show()