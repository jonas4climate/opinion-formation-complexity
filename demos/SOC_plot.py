"""
Loads data for SOC and simulates!
"""

import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt
import csv
import powerlaw
from scipy.ndimage import measurements

# Retrieve the sim data
#data = np.fromfile('./figures/SOC_sim.npy')#, dtype=float)

opinions = np.load('./figures/SOC_sim_opinions.npy')
cluster_sizes = np.load('./figures/SOC_sim_cluster_sizes.npy')


"""
data = []
reader = csv.DictReader(open('./figures/SOC_sim.csv'))
for row in reader:
    print(row)
    data.append(dict(row))

data = data[0]
"""


#opinions = data_copy['opinions'][5].copy()
#opinions[opinions == 1] = 0
#opinions[opinions == -1] = 1
#print(opinions)

params = []
reader = csv.DictReader(open('./figures/SOC_params.csv'))
for row in reader:
    print(row)
    params.append(dict(row))
    # profit !


params = params[0]

TIMESTEPS = int(params['TIMESTEPS'])

GRIDSIZE_X = int(params['GRIDSIZE_X'])
GRIDSIZE_Y = int(params['GRIDSIZE_Y'])
p = float(params['P_OCCUPATION'])
p_1 = float(params['P_OPINION_1'])
critical_temperature = float(params['CRITICAL_TEMPERATURE'])
H = float(params['H'])
BETA_PEOPLE = float(params['BETA_PEOPLE'])
BETA_LEADER = float(params['BETA_LEADER'])
INFLUENCE_DISTRIBUTION_MEAN = float(params['S_MEAN'])
INFLUENCE_LEADER = float(params['INFLUENCE_LEADER'])

N = int(params['N'])

"""
params_used = {
               'CRITICAL_TEMPERATURE':critical_temperature,
               'TIMESTEPS':TIMESTEPS,
               'S_LEADER':INFLUENCE_LEADER,
               'GRIDSIZE_X': GRIDSIZE_X,
               'GRIDSIZE_Y': GRIDSIZE_Y,
               'TIMESTEPS':TIMESTEPS,
               'BETA':BETA_PEOPLE,
               'BETA_LEADER':BETA_LEADER,
               'INFLUENCE_LEADER':INFLUENCE_LEADER,
               'H':H,
               'P_OCCUPATION':p,
               'P_OPINION_1':p_1,
               'a_0':a_0,
               'S_MEAN':INFLUENCE_DISTRIBUTION_MEAN
               }
"""



# Make a copy to avoid overwriting the data
#data_copy = data.copy()


# Step 4 - Plot opinion change to see if clusters do form
#model.plot_opinion_grid_at_time_t(data_copy,5) # This creates a video in the folder figures!

temperature = critical_temperature


# Do video
#model = ca.CA(GRIDSIZE_X, GRIDSIZE_Y, p, p_1, critical_temperature, H, BETA_PEOPLE, BETA_LEADER, INFLUENCE_DISTRIBUTION_MEAN, INFLUENCE_LEADER, 'euclidean', 'linear', 1, 'uniform', True)
#model.plot_opinion_grid_evolution(data_copy,save=True)


#data_copy = data.copy()
#sim_data = data_copy.copy()

total_unique = np.linspace(1,N,N)
total_counts = np.zeros(N).astype(int)

#print('Total counts',total_counts)

for time_step in range(TIMESTEPS):
    print('time_step',time_step)
    data_t = opinions[time_step, :, :].copy()
    data_t[data_t == 1] = 0
    data_t[data_t == -1] = 1

    lw, num_cluster = measurements.label(data_t)
    areas = measurements.sum(data_t, lw, index=range(lw.max() + 1))
    unique, counts = np.unique(areas, return_counts=True)

    unique = unique.astype(int)
    counts = counts.astype(int)

    # Update the total_counts at those locations
    #np.take(total_counts, unique)
    # https://numpy.org/doc/stable/reference/generated/numpy.put.html
    np.put(total_counts,unique,np.take(total_counts, unique) + counts) # Should be counts+previous value

    #np.place(total_counts, unique, counts)

    #print(unique, counts)
    #print(total_counts)

    # Add counts of every unique value
    #for index in unique:

        #print('Unique',int(index))
        #print('Test',total_counts[int(index)])
        #print('Test2',counts[int(index)])

        #total_counts[int(index)] += counts[int(index)]


    #plt.scatter(unique,counts,facecolors='none', edgecolors='blue')
    #print(unique,counts)

#print(total_counts)




d=total_counts
fit = powerlaw.Fit(np.array(d)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
ax = fit.plot_pdf( color= 'b')

ax.set_title('PDF of CA at critical temperature')
ax.set_xlabel('size of connected nodes')
ax.set_ylabel('p(connected nodes)')



plt.grid()
plt.tight_layout()
plt.savefig('./figures/SOC_plot.png')

plt.show()



print(fit)

print('alpha= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)

