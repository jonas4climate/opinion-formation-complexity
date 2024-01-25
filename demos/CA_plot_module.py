"""
Our own module to plot the 2D CA

"""

import numpy as np
import CA_module as ca
import matplotlib.pyplot as plt

def diagram(R,BETA,H):
    fig, ax = plt.subplots()

    S_L_min = ca.minimun_leader_strength(R,BETA,H)
    S_L_max = ca.maximun_leader_strength(R,BETA,H)
    cluster_min = ca.a(R,BETA,H,S_L_min)
    cluster_max = ca.a(R,BETA,H,S_L_max)

    xmin,xmax = 0,600
    ymin,ymax = 0,22.5


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
    y = ca.a_positive(R,BETA,H,x)
    ax.plot(x,y,c='black',linestyle='--')

    # Parabola under arm
    x = np.linspace(S_L_min, S_L_max, 100)
    y2 = ca.a_negative(R,BETA,H,x)
    ax.plot(x,y2,c='black',linestyle='--')

    # Vertical line
    x = np.linspace(S_L_min, S_L_max, 100)
    ax.vlines(x=S_L_min, ymin=0, ymax=cluster_min[0], colors='gray', ls='dotted', lw=1)

    # Complete consensus line
    x_cons = np.linspace(xmin, xmax, 100)
    y_cons = np.ones(100)*R
    ax.plot(x_cons,y_cons,c='black',linestyle='-')

    # Title
    ax.set_title(f'R={R}, Beta={BETA}, H={H}')
    ax.set_ylabel('a')
    ax.set_xlabel('S_L')

    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])

    plt.grid()
    #plt.legend()
    plt.tight_layout()


    return fig, ax