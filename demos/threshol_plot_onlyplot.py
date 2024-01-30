"""Read files with results
Plot with std dev"""
import ca.cellular_automata as ca
import numpy as np
import matplotlib.pyplot as plt
import csv


params = []
reader = csv.DictReader(open('./figures/t_threshold_plot_params.csv'))
for row in reader:
    print(row)
    params.append(dict(row))
    # profit !

print(params)

params = params[0]

R = params['R']
T_MAX
SIMS_PER_T_VALUE
THRESHOLD

temperatures
p_overcoming_leader
means
std_devs




plt.figure()
plt.plot(temperatures,p_overcoming_leader,lw=2,c='blue')
plt.fill_between(temperatures, np.array(means)-np.array(std_devs), np.array(means)+np.array(std_devs), alpha=0.3)


plt.xlim([0,T_MAX])
plt.ylim([0,1])

plt.suptitle('Effect of temperature on overcoming leader consensus')
plt.title(f'R={R}, {SIMS_PER_T_VALUE} runs/T, T_MAX={T_MAX},THRESHOLD={THRESHOLD}')
plt.xlabel('Temperature')
plt.ylabel('p(Overcoming leader)')

plt.grid()
plt.tight_layout()


plt.savefig('./figures/t_threshold_plot_FIX.png')


plt.show()