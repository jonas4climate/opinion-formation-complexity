"""
2D CA Simulation using the CA_module.py

"""

import CA_module as ca
import numpy as np
import matplotlib.pyplot as plt

################################

GRIDSIZE_X,GRIDSIZE_Y = 5,21
TIMESTEPS = 30
TEMPERATURE = 40
BETA_PEOPLE = 1
BETA_LEADER = 1
H = 0
p = 1
p_1 = 0
a_0 = 1 # Size of innitial cluster around leader

INFLUENCE_LEADER = 50   # Leader influence
INFLUENCE_DISTRIBUTION_MEAN = 1



################################

if TEMPERATURE == 0:
    expect_cluster = ca.analytical_expect_clusters(
        GRIDSIZE_X/2, BETA_PEOPLE, H, INFLUENCE_LEADER)
    print('Expect clusters?', expect_cluster)

model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, TEMPERATURE, BETA_LEADER, BETA_PEOPLE, H, p, p_1, INFLUENCE_LEADER, INFLUENCE_DISTRIBUTION_MEAN, ca.euclidean_distance, ca.prob_dist_influence_people)
data = model.evolve(5)

model.reset()
data = model.evolve(TIMESTEPS)
model.plot_opinion_grid_at_time_t(data, 10)
plt.show(block=True)


################################

################################

# Plot
# cpl.plot2d_animate(simulation,interval=250)


# Plot influence grid
# TODO: Make grid out of node_influences
# plt.imshow(node_influences, cmap='seismic', interpolation='nearest')
# plt.colorbar()
# plt.title('Influences')
# plt.show(block=False)

# Colors look up table, so we will always map colors correctly


fig, ax = plt.subplots()


for time_step in range(TIMESTEPS):

    ## PLOT OPINION EVOLUTION IN TIME
    # Retrieve plot
    grid_t = simulation[time_step, :, :]
    cluster_size = cluster_sizes[time_step]

    # Plot it
    im = ax.imshow(grid_t, cmap='seismic',
                   interpolation='nearest', vmin=-1, vmax=1)
    # Text of opinion
    #for n in range(N):
    #    x, y = int(node_coordinates[n, 0]), int(node_coordinates[n, 1])
    #    text = ax.text(
    #        y, x, str(int(grid_t[x, y])), ha="center", va="center", color="w",fontsize=4)
    ax.set_title(f'Frame:{(time_step+1)}/{TIMESTEPS},\n T={TEMPERATURE}, H={H}, B={BETA_PEOPLE}, Bl={BETA_LEADER}, sL={INFLUENCE_LEADER},c_radius={int(cluster_size)}')
    
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)
    ax.clear()

    # plt.close()

# Keep last frame until manually closing it
plt.show(block=True)

# Phase animation
fig, ax = ca.plot_diagram(GRIDSIZE_X/2,BETA_PEOPLE,H)

R = GRIDSIZE_X/2
S_L_min = ca.minimun_leader_strength(R,BETA_PEOPLE,H)
S_L_max = ca.maximun_leader_strength(R,BETA_PEOPLE,H)
cluster_min = ca.a(R,BETA_PEOPLE,H,S_L_min)
cluster_max = ca.a(R,BETA_PEOPLE,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5


for time_step in range(TIMESTEPS):

    # Parabola critical points
    ax.scatter(S_L_min,cluster_min[0],c='black')
    ax.scatter(S_L_min,cluster_min[1],c='black')
    ax.scatter(S_L_max,cluster_max[0],c='black')

    # Floor
    x_floor = np.linspace(0, S_L_min, 100)
    y_floor = np.zeros(100)
    ax.plot(x_floor,y_floor,c='black',linestyle='--')

    # Parabola top arm
    x = np.linspace(0, S_L_max, 100)
    y = ca.a_positive(R,BETA_PEOPLE,H,x)
    ax.plot(x,y,c='black',linestyle='--')

    # Parabola under arm
    x = np.linspace(S_L_min, S_L_max, 100)
    y2 = ca.a_negative(R,BETA_PEOPLE,H,x)
    ax.plot(x,y2,c='black',linestyle='--')

    # Vertical line
    x = np.linspace(S_L_min, S_L_max, 100)
    ax.vlines(x=S_L_min, ymin=0, ymax=cluster_min[0], colors='gray', ls='dotted', lw=1)

    # Complete consensus line
    x_cons = np.linspace(xmin, xmax, 100)
    y_cons = np.ones(100)*R
    ax.plot(x_cons,y_cons,c='black',linestyle='-')

    # Title
    ax.set_title(f'Frame:{(time_step+1)}/{TIMESTEPS}, R={R}, Beta={BETA_PEOPLE}, H={H}')
    ax.set_ylabel('a')
    ax.set_xlabel('S_L')

    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])

    # Current leader influence!!!
    
    ax.vlines(x=INFLUENCE_LEADER, ymin=0, ymax=ymax, colors='gray', ls='dashed', lw=1)


    # Current cluster !!!!!
    cluster_size = cluster_sizes[time_step]
    ax.scatter(INFLUENCE_LEADER,cluster_size)
    
    
    plt.grid()
    plt.tight_layout()
    plt.pause(0.2)
    plt.show(block=False)
    ax.clear()


plt.show(block=True)