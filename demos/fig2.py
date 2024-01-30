""""
Plot graph 2 of paper
"""

import numpy as np
import demos.ca.cellular_automata as ca
import matplotlib.pyplot as plt
from scipy.special import ellipeinc

R = 41/2 #5/2
BETA = 1
H = 0
S_L = 400

a1,a2 = ca.a(R,BETA,H,S_L)

#print("a1a2",a1,a2)

plt.suptitle(' Social impact I as a function of distance d to the leader')
plt.title(f'R={R}, Beta={BETA}, H={H}')
plt.xlabel('d')
plt.ylabel('I')

xmin,xmax = 0,20
ymin,ymax = -600,0
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])


# d > a
x1 = np.linspace(a2, R, 100)
y1 = ca.impact_out(S_L,a2,R,x1,BETA)
plt.plot(x1,y1,c='black',linestyle='--')

# d < a
x2 = np.linspace(0, a2, 100)
y2 = ca.impact_in(S_L,a2,R,x2,BETA)
plt.plot(x2,y2,c='black',linestyle='--')

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()