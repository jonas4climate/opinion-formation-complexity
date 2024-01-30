""""
Plot fig.3 of the paper
Used the equation (9) and (10), assuming that the mean cluster radius a_T calculated by the proportion of number of cells present 1, which means:
a_T_proportion = (number of cells have 1) / area of the circle with radius R(R=GRIDSIZE_X/2)

Added a new function for calculate the mean cluster radius

Caution: Just focus on the gird of final timestep 

TODO: Still need to be tested for the optimum parameter
TODO: Speed up the simulation (if can)
"""
import numpy as np
import ca.cellular_automata as ca
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

# Parameters
GRIDSIZE_X,GRIDSIZE_Y = 45,45
TIMESTEPS = 10
BETA = 1
BETA_LEADER = 1
H = 0
P_OCCUPATION = 1
P_OPINION_1 = 0

S_LEADER = 400   
S_MEAN = 1

T_MAX = 40


TEMPERATURES=np.linspace(0,T_MAX,10)
N_SIMS = 10    # Run times
Mean_cluster_radius = []

def simulate_single_run(t):
    model = ca.CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=t, beta=BETA, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN, show_tqdm=False)
    data = model.evolve(TIMESTEPS)
    simulation_final = data['opinions'][TIMESTEPS-1]
    a_T = model.mean_cluster_radius()
    return a_T

# If main process and not pool processes
if __name__ == '__main__':

    with Pool() as pool:
        pbar = tqdm(TEMPERATURES, total=len(TEMPERATURES)*N_SIMS, desc="Running simulation in batches", unit='sim')
        for t in pbar:
            a_Ts = pool.map(simulate_single_run, [t]*N_SIMS)
            Mean_cluster_radius.append(np.mean(a_Ts)) 
            pbar.update(N_SIMS)     
        print("Mean_cluster_radius",Mean_cluster_radius)


    # Plotting
    plt.suptitle(' Mean cluster radius a vs. temperature T')
    plt.title(f'S_L={S_LEADER},simulation={N_SIMS},GRIDSIZE={GRIDSIZE_X},H={H}')
    plt.xlabel('T')
    plt.ylabel('a(T)')

    xmin,xmax = 0,T_MAX
    ymin,ymax = 0,10
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.plot(TEMPERATURES,Mean_cluster_radius,marker="o")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()