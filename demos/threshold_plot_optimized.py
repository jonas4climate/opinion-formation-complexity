"""

Creates a threshold plot in terms of temperature:
probability of overcoming leader's opinion as temperature increases

note that for this scenario, everybody has leaders opinion!

p_overcoming_leader = f(TEMPERATURE)

"""

import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt


from multiprocessing import Pool
from tqdm import tqdm

################################

NUMBER_OF_T_VALUES_TO_TEST = 20
SIMS_PER_T_VALUE = 20
T_MAX = 200

S_LEADER = 100             # May need to tweak this to ensure we are on cluster region!
################################

# Todo this we first get parameters that for T=0
# ensure either unification with leader

################################

GRIDSIZE_X,GRIDSIZE_Y = 21,21
TIMESTEPS = 10
#TEMPERATURE = 0
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 1 # In this scenario everybody believes the leader at start
a_0 = 1

S_MEAN = 1

################################

R = GRIDSIZE_X/2
THRESHOLD = int(np.sqrt((R**2)/2)) #5 # Maximun leader cluster radius that is not considered opinion overcomming



S_L_min = ca.minimun_leader_strength(R,BETA,H)
S_L_max = ca.maximun_leader_strength(R,BETA,H)
cluster_min = ca.a(R,BETA,H,S_L_min)
cluster_max = ca.a(R,BETA,H,S_L_max)
xmin,xmax = 0,2*S_L_max
ymin,ymax = 0,22.5

################################

# Everything that the pool needs should be here
# Only one per t?
def simulate_single_run(t):
    model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=t, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN, show_tqdm=False)
    data = model.evolve(TIMESTEPS)
    last_cluster_size = data['cluster_sizes'][-1]

    return last_cluster_size



if __name__ == '__main__':
    # Spawn all pools


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

    # Do all the ones for a specific temperature!
    def simulate(index):
            TEMP = temperatures[index]
            leader_overcomed = 0

            for sim in range(SIMS_PER_T_VALUE):
                model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)
                data = model.evolve(TIMESTEPS)
                last_cluster_size = data['cluster_sizes'][-1]

                if last_cluster_size <= THRESHOLD:
                    leader_overcomed += 1

            return leader_overcomed / SIMS_PER_T_VALUE





    with Pool() as pool:
        pbar = tqdm(temperatures, total=len(temperatures)*SIMS_PER_T_VALUE, desc="Running simulation in batches", unit='sim')
        for t in pbar:
            # Get all last cluster sizes for all sims with same temperautre
            #last_cluster_sizes = list(pool.map(simulate_single_run, [t]*NUMBER_OF_T_VALUES_TO_TEST))
            last_cluster_sizes = list(pool.map(simulate_single_run, [t]*SIMS_PER_T_VALUE))
        
            # Debugging...
            print('Last cluster sizes',last_cluster_sizes,THRESHOLD)
            for element in last_cluster_sizes:
                # Its a dict with key 'cluster_sizes'
                c_sizes = element['cluster_']
                print('Element\n',element)
            
            # Turn them into an array
            last_cluster_sizes = np.array(last_cluster_sizes)

            # Get the p_overcoming_leader out of last_cluster_sizes
            # Should be the ammount of elements <= THRESHOLD, then / SIMS_PER_T_VALUE            
            prob_t = np.count_nonzero(last_cluster_sizes < THRESHOLD) / SIMS_PER_T_VALUE
    
            p_overcoming_leader[t] = prob_t

            # Update the progress bar
            pbar.update(SIMS_PER_T_VALUE) 
    
    #num_processes = 4
    #pool = Pool(processes=num_processes)
    #results = pool.map(simulate, range(NUMBER_OF_T_VALUES_TO_TEST))

    #p_overcoming_leader = np.array(results)

    ################################

    # Plot the threshold phenomena

    plt.figure()
    plt.plot(temperatures,p_overcoming_leader)

    plt.xlim([0,T_MAX])
    plt.ylim([0,1])

    plt.suptitle('Effect of temperature on overcoming leader consensus')
    plt.title(f'R={R}, {SIMS_PER_T_VALUE} runs/T, T_MAX={T_MAX},THRESHOLD={THRESHOLD}')
    plt.xlabel('Temperature')
    plt.ylabel('p(Overcoming leader)')

    plt.grid()
    plt.tight_layout()

    plt.show()