"""
Plots graph 1 of paper analitically
"""

import numpy as np
import CA_module as ca
import matplotlib.pyplot as plt

R = 20 #5/2
BETA = 1
H = 0

S_L_min = ca.minimun_leader_strength(R,BETA,H)
S_L_max = ca.maximun_leader_strength(R,BETA,H)

cluster_min = ca.a(R,BETA,H,S_L_min)
cluster_max = ca.a(R,BETA,H,S_L_max)

print(f'R,beta,H: [{R},{BETA},{H}]')
print(f'Range of SL that gives clusters: [{S_L_min},{S_L_max}]')

#S_L = int(input('Give SL value: '))
#a1,a2 = ca.a(R,BETA,H,S_L)
#print(f'Expected cluster size: [{a1},{a2}]')


# Plot a=f(S_L)

plt.suptitle('Effect of leader influence on opinion cluster size')
plt.title(f'R={R}, Beta={BETA}, H={H}')
plt.ylabel('a')
plt.xlabel('S_L')

xmin,xmax = 0,600
ymin,ymax = 0,22.5
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])

#plt.xticks(list(plt.xticks()[0]) + [S_L_min,S_L_max])

# Parabolla limits
plt.scatter(S_L_min,cluster_min[0],c='black')
plt.scatter(S_L_min,cluster_min[1],c='black')
plt.scatter(S_L_max,cluster_max[0],c='black')

# Parabola top arm
x = np.linspace(0, S_L_max, 100)
y = ca.a_positive(R,BETA,H,x)
plt.plot(x,y,c='black',linestyle='--')

# Parabola under arm
x = np.linspace(S_L_min, S_L_max, 100)
y2 = ca.a_negative(R,BETA,H,x)
plt.plot(x,y2,c='black',linestyle='--')

# Vertical line
x = np.linspace(S_L_min, S_L_max, 100)

plt.vlines(x=S_L_min, ymin=0, ymax=cluster_min[0], colors='gray', ls='dotted', lw=1)


# Floor
x_floor = np.linspace(0, S_L_min, 100)
y_floor = np.zeros(100)
plt.plot(x_floor,y_floor,c='black',linestyle='--')

# Complete consensus
x_cons = np.linspace(xmin, xmax, 100)
y_cons = np.ones(100)*R
plt.plot(x_cons,y_cons,c='black',linestyle='-')


# Stable configuration  
#x = np.linspace(0, S_L_max, 100)
#y3 = np.ones(100)*ca.a_stable(R,BETA,H,x)
#print(y3)
#plt.plot(x,y3,c='black',linestyle='--')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()