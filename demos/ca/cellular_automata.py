import numpy as np
from math import floor
from scipy.special import ellipeinc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from tqdm import tqdm
import logging
from logging import warning, error, info, debug

logging.basicConfig(level=logging.INFO)


def distance_metric(x0, y0, x1, y1, type='euclidean'):
    if type == 'euclidean':
        return np.sqrt((x1-x0)**2 + (y1-y0)**2)
    elif type == 'von_neumann':
        return abs(x1-x0) + abs(y1-y0)
    elif type == 'moore':
        return max(abs(x1-x0), abs(y1-y0))


def prob_dist_s_people(mean, type='uniform'):

    # Probability distribution for the node influence
    if type == 'uniform':
        return np.random.uniform(0, 2*mean)

    if type == 'normal':
        return np.abs(np.random.normal(mean, scale=1))

    if type == 'exponential':
        return np.random.exponential(mean)


def g(distance_ij, type='linear', c=1):
    """
    Define function g based on distance distance_ij and constant c
    """
    if type == 'linear':
        return c * distance_ij

    if type == 'exponential':
        return c * np.exp(distance_ij)

    if type == 'power':
        return distance_ij**c

    if type == 'logarithmic':
        return c * np.log(distance_ij)


class CA(object):

    def __init__(self, gridsize_x, gridsize_y, p_occupation, p_opinion_1, temp, h, beta, beta_leader, s_mean, s_leader, dist_func='euclidean', dist_scaling_func='linear', dist_scaling_factor=1, s_prob_dist_func='uniform', show_tqdm=True):
        # Parameters directly provided
        self.gridsize_x, self.gridsize_y = gridsize_x, gridsize_y
        self.p_occupation = p_occupation
        self.p_opinion_1 = p_opinion_1
        self.temp = temp
        self.h = h
        self.beta = beta
        self.beta_leader = beta_leader
        self.s_mean = s_mean
        self.s_leader = s_leader

        # Functions passed
        self.d = lambda x0, y0, x1, y1: distance_metric(
            x0, y0, x1, y1, type=dist_func)
        self.g = lambda d: g(d, type=dist_scaling_func, c=dist_scaling_factor)
        self.q = lambda mean: prob_dist_s_people(mean, type=s_prob_dist_func)

        # CA itself
        self.starting_grid = self.__gen_initial_opinion_grid()
        self.opinion_grid = self.starting_grid.copy()
        self.N = self.__gen_number_of_nonempty_nodes_in_grid()

        # Utility variables for optimization
        self.__node_coords = self.__gen_array_node_coord_tuples()
        self.__leader_node_coord_idx = self.__gen_leader_coord_index()
        self.distance_matrix = self.__gen_distance_matrix()
        self.beta_matrix = self.__gen_beta_matrix(beta, beta_leader)
        self.__node_influences = self.__gen_node_influences(s_leader)

        # Extra
        self.show_tqdm = show_tqdm

    def __gen_initial_opinion_grid(self):
        assert self.gridsize_x % 2 == 1 and self.gridsize_y % 2 == 1, 'Gridsize must be odd for central leader placement'

        grid = np.zeros((self.gridsize_x, self.gridsize_y))

        center_x = int((self.gridsize_x-1)/2)
        center_y = int((self.gridsize_y-1)/2)

        # Assign nonzero values to individuals outside the grid
        R = self.gridsize_x/2
        for x_idx in range(self.gridsize_x):
            for y_idx in range(self.gridsize_y):
                # Only individuals inside the circle are considered
                if self.d(x_idx, y_idx, center_x, center_y) <= R:
                    r = np.random.rand(1)
                    if r < self.p_occupation:
                        # Get -1 or 1 with p1
                        r = np.random.rand(1)
                        grid[x_idx, y_idx] = 1 if r < self.p_opinion_1 else -1

        # Add leader in center with opinion 1
        grid[center_x, center_y] = 1

        return grid

    def __gen_number_of_nonempty_nodes_in_grid(self):
        N = np.count_nonzero(self.opinion_grid)
        return N

    def __gen_array_node_coord_tuples(self):
        N = self.N
        node_coordinates = np.zeros((N, 2))

        n = 0
        for x, y in np.ndindex(self.opinion_grid.shape):
            if self.opinion_grid[x, y] != 0:
                node_coordinates[n, 0] = x
                node_coordinates[n, 1] = y
                # Update the index to save in the right row of node_coordinates
                n += 1
        return node_coordinates

    def __gen_leader_coord_index(self):
        center_x = int((self.gridsize_x-1)/2)
        center_y = int((self.gridsize_y-1)/2)

        N = self.N

        for n in range(N):
            if self.__node_coords[n, 0] == center_x and self.__node_coords[n, 1] == center_y:
                return n

        raise ValueError('Leader not found!')

    def __gen_beta_matrix(self, beta_people, beta_leader):
        beta_matrix = np.full(self.N, beta_people)
        beta_matrix[self.__leader_node_coord_idx] = beta_leader
        return beta_matrix

    def __gen_distance_matrix(self):
        # Distance matrix (only needed to compute it once!)
        # It is a 3D matrix of size GRIDSIZE_X,GRIDSIZE_,N
        # Each submatrix is the distance grid from each node
        # TODO (Secondary): Optimize this knowing the distance matrix is symetric
        node_coords = self.__node_coords
        N = node_coords.shape[0]
        distance_matrix = np.zeros((N, N))

        # For all nodes
        for n_i in range(N):
            # Compute distance to all other nodes
            for n_j in range(N):
                # Retrieve those values
                x0 = node_coords[n_i, 0]
                y0 = node_coords[n_i, 1]
                x1 = node_coords[n_j, 0]
                y1 = node_coords[n_j, 1]

                # Update the distance
                distance_matrix[n_i, n_j] = self.d(x0, y0, x1, y1)

        # Extra, set closest neighboor distance to 1
        # And other distances to quadratic
        # Iterate rows

        return distance_matrix

    def __gen_node_influences(self, s_L):
        # Influence (computed once!)
        node_influences = np.zeros(self.N)

        for i in range(self.N):
            node_influences[i] = s_L if i == self.__leader_node_coord_idx else self.q(
                self.s_mean)

        return node_influences

    def __cluster_size_leader(self):
        # Find opinion of leader!
        gridsize_x, gridsize_y = self.opinion_grid.shape
        center_x = int((gridsize_x-1)/2)
        center_y = int((gridsize_y-1)/2)
        leader_opinion = self.opinion_grid[center_x, center_y]

        # Get distance of nodes to leader from distance matrix!!!
        leader_distance_matrix = self.distance_matrix[self.__leader_node_coord_idx, :]

        # Cluster has radius r if all nodes at a smaller distance than r
        # to the center have the same opinion as the leader
        # Start with radius 0
        c_radius = 0.5
        max_c_radius = floor(max(gridsize_x, gridsize_y)/2)
        consulted_nodes = np.array([])

        while c_radius < max_c_radius:
            # Find all nodes in distance_matrix closer that c_radius
            nodes = np.where(leader_distance_matrix <= c_radius)[0]

            # print('Nodes',nodes,consulted_nodes.astype(int))
            # nodes = np.where(nodes != consulted_nodes)[0]

    #        print(np.where(nodes != consulted_nodes))
            # Remove consulted nodes from nodes
            # if c_radius>0:
            #    nodes = np.delete(nodes, consulted_nodes.astype(int))
            # print('nodes',nodes[0])

            for n in nodes:
                nx, ny = self.__node_coords[n, 0], self.__node_coords[n, 1]
                if int(self.opinion_grid[int(nx), int(ny)]) != int(leader_opinion):
                    # If somebody has different opinion than leader, then we dont have cluster
                    # print('NOOO')
                    return c_radius

            # TODO: Remove consulted nodes
            # consulted_nodes = np.append(consulted_nodes, nodes, axis=0)
            # print(consulted_nodes)

            c_radius += 1

        # Turn radius to size
        # c_size = np.pi*c_radius**2

        return c_radius  # ,c_size

    def __update_opinions(self):
        grid = self.opinion_grid.copy()  # to ensure we don't modify the input grid
        # Update opinion of each node
        for i in range(self.N):

            # First compute impact
            # Retrieve node opinion from grid
            # and node influence from matrix
            i_x, i_y = int(self.__node_coords[i, 0]), int(
                self.__node_coords[i, 1])

            sigma_i = grid[i_x, i_y]
            s_i = self.__node_influences[i]

            # Compute sum by looping over other nodes
            summation = 0
            for j in range(self.N):
                if j != i:
                    # Retrieve their opinion and influence and distance
                    # And add the term
                    j_x, j_y = int(self.__node_coords[j, 0]), int(
                        self.__node_coords[j, 1])

                    sigma_j = grid[j_x, j_y]
                    s_j = self.__node_influences[j]
                    d_ij = self.distance_matrix[i, j]

                    # Compute the function of the distance
                    g_d_ij = self.g(d_ij)

                    summation += (s_j * sigma_i * sigma_j)/(g_d_ij)

            # Combine to get social impact
            I_i = -s_i*self.beta_matrix[i] - sigma_i*self.h - summation

            # Update opinion
            if self.temp == 0:
                # Compute probability
                sigma_i = -np.sign(I_i * sigma_i)
            else:
                # Compute probability of change and update if neccesary
                probability_staying = (np.exp(-I_i / self.temp)) / \
                    (np.exp(-I_i / self.temp) + np.exp(I_i / self.temp))
                opinion_change = bool(np.random.rand(1) > probability_staying)

                if opinion_change:
                    sigma_i = -sigma_i

            # Save the new opinion on the grid
            grid[i_x, i_y] = sigma_i

        # Save the updated grid of the time step
        # simulation[time_step+1,:,:] = grid
        return grid

    # new function for fig.3
    def mean_cluster_radius(self):

        center_x = (self.gridsize_x - 1) // 2
        center_y = (self.gridsize_y - 1) // 2

        leader_opinion = self.opinion_grid[center_x, center_y]

        R = self.gridsize_x // 2
        circle_area = np.pi * R**2

        # calculate the number of cells with opinion +1
        positive_opinions = np.sum(self.opinion_grid == leader_opinion)

        # calculate the mean cluster radius
        mean_cluster_radius = np.sqrt(positive_opinions / circle_area * R**2)
        return mean_cluster_radius
    #######

    def evolve(self, timesteps):
        opinion_grid_history = np.ndarray(
            (timesteps+1, self.gridsize_x, self.gridsize_y))
        cluster_sizes = np.zeros(timesteps+1)

        opinion_grid_history[0, :, :] = self.opinion_grid
        cluster_sizes[0] = self.__cluster_size_leader()

        # TODO: should be timesteps, code needs to be adjusted everywhere
        timesteps_iter = tqdm(range(timesteps)) if self.show_tqdm else range(timesteps)
        for time_step in timesteps_iter:
            grid = self.__update_opinions()
            cluster_sizes[time_step+1] = self.__cluster_size_leader()
            opinion_grid_history[time_step+1, :, :] = grid
            self.opinion_grid = grid

        data = {'opinions': opinion_grid_history,
                'cluster_sizes': cluster_sizes}
        return data

    def reset(self):
        if np.all(self.starting_grid == self.opinion_grid):
            warning('Grid hasn\'t changed since initialization, no need to reset.')
            return

        self.opinion_grid = self.starting_grid.copy()
        return

    def plot_opinion_grid_at_time_t(self, data, t, save=False):
        """
        Plot the opinion grid at timestep t from data returned by `evolve()`
        """
        plt.figure(figsize=(6, 6), layout='tight')
        opinion_history = data['opinions']
        plt.imshow(opinion_history[t], vmin=-1, vmax=1)
        plt.title(
            f"Opinion grid at $t={t}$ ($T={self.temp}$, $s_l={self.s_leader}$, $\\hat{{s}}$={self.s_mean}, $\\beta$={self.beta}, $p_{{occ}}$={self.p_occupation}, $p_{{1}}$={self.p_opinion_1})")
        plt.axis(False)
        plt.grid(False)
        if save:
            plt.savefig(
                f'figures/{self.gridsize_x}x{self.gridsize_y}_opinion_grid_t={t}.png', dpi=300)

    def plot_opinion_grid_evolution(self, data, viz_range=None, interval=250, save=False):
        """
        Plot the evolution of the opinion grid from data returned by `evolve()`
        """
        plt.figure(figsize=(6, 6), layout='tight')
        opinion_history = data['opinions']

        def update(t):
            plt.clf()
            plt.imshow(opinion_history[t], vmin=-1, vmax=1)
            plt.axis(False)
            plt.grid(False)
            plt.title(
                f"Opinion grid at $t={t}$ ($T={self.temp}$, $s_l={self.s_leader}$, $\\hat{{s}}$={self.s_mean}, $\\beta$={self.beta}, $p_{{occ}}$={self.p_occupation}, $p_{{1}}$={self.p_opinion_1})")
            return plt

        if viz_range is None:
            anim = FuncAnimation(plt.gcf(), update, frames=range(
                0, opinion_history.shape[0]), interval=interval)
        else:
            anim = FuncAnimation(plt.gcf(), update, frames=range(
                *viz_range), interval=interval)
        if save:
            anim.save(
                f'figures/{self.gridsize_x}x{self.gridsize_y}_opinion_grid_evolution.mp4', dpi=300)

# do not know which a to use but use a1 first
# Inside and outside impact


def impact_in(s_l, a, r, distance_to_leader, beta):

    term1 = - s_l / distance_to_leader
    term2 = - 8 * a * ellipeinc(np.pi / 2, (distance_to_leader / a)**2)
    term3 = 4 * r * ellipeinc(np.pi / 2, (distance_to_leader / r)**2)
    term4 = 2 * np.sqrt(np.pi)

    result = term1 + term2 + term3 + term4 - beta

    return result


def impact_out(s_l, a, r, distance_to_leader, beta):
    term1 = s_l / distance_to_leader
    term2 = 8 * a * \
        ellipeinc(np.arcsin(a / distance_to_leader),
                  (a / distance_to_leader)**2)
    term3 = -4 * r * ellipeinc(np.pi / 2, (distance_to_leader / r)**2)
    term4 = 2 * np.sqrt(np.pi)

    result = term1 + term2 + term3 + term4 - beta

    return result


def plot_diagram(r, beta, h):
    fig, ax = plt.subplots()

    S_L_min = minimun_leader_strength(r, beta, h)
    S_L_max = maximun_leader_strength(r, beta, h)
    cluster_min = a(r, beta, h, S_L_min)
    cluster_max = a(r, beta, h, S_L_max)

    xmin, xmax = 0, 600
    ymin, ymax = 0, 22.5

    # Parabola critical points
    ax.scatter(S_L_min, cluster_min[0], c='black')
    ax.scatter(S_L_min, cluster_min[1], c='black')
    ax.scatter(S_L_max, cluster_max[0], c='black')

    # Floor
    x_floor = np.linspace(0, S_L_min, 100)
    y_floor = np.zeros(100)
    ax.plot(x_floor, y_floor, c='black', linestyle='--')

    # Parabola top arm
    x = np.linspace(0, S_L_max, 100)
    y = a_positive(r, beta, h, x)
    ax.plot(x, y, c='black', linestyle='--')

    # Parabola under arm
    x = np.linspace(S_L_min, S_L_max, 100)
    y2 = a_negative(r, beta, h, x)
    ax.plot(x, y2, c='black', linestyle='--')

    # Vertical line
    x = np.linspace(S_L_min, S_L_max, 100)
    ax.vlines(x=S_L_min, ymin=0,
              ymax=cluster_min[0], colors='gray', ls='dotted', lw=1)

    # Complete consensus line
    x_cons = np.linspace(xmin, xmax, 100)
    y_cons = np.ones(100)*r
    ax.plot(x_cons, y_cons, c='black', linestyle='-')

    # Title
    ax.set_title(f'R={r}, Beta={beta}, H={h}')
    ax.set_ylabel('a')
    ax.set_xlabel('S_L')

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    plt.grid()
    # plt.legend()
    plt.tight_layout()

    return fig, ax


def update_basics_diagram(fig, ax):

    return fig, ax


def analytical_expect_clusters(r, beta, h, s_l):
    # Ensure both solutions are > 0

    #print('First half (2*pi*R-sqrt(pi)+beta-h)^2:',
    #      (2*np.pi*r - np.sqrt(np.pi) + beta - h)**2)
    #print('Second half (32*s_l):', 32*s_l)

    condition_1 = bool(
        (2*np.pi*r - np.sqrt(np.pi) + beta - h)**2 - 32*s_l >= 0)
    condition_2 = bool(
        (2*np.pi*r - np.sqrt(np.pi) - beta - h)**2 - 32*s_l >= 0)
    return condition_1 and condition_2


def a(r, beta, h, s_l):
    """
    Calculate the cluster size in determistic model case assuming g(r) = r and mean s = 1
    """
    a_1 = 1/16*(2*np.pi*r - np.sqrt(np.pi) + beta - h +
                np.sqrt((2*np.pi*r - np.sqrt(np.pi) + beta - h)**2 - 32 * s_l))
    a_2 = 1/16*(2*np.pi*r - np.sqrt(np.pi) - beta - h -
                np.sqrt((2*np.pi*r - np.sqrt(np.pi) - beta - h)**2 - 32 * s_l))

    return a_1, a_2


def a_positive(r, beta, h, s_l):
    # + beta + sqrt (stable cluster)
    return 1/16*(2*np.pi*r - np.sqrt(np.pi) + beta - h + np.sqrt((2*np.pi*r - np.sqrt(np.pi) + beta - h)**2 - 32 * s_l))


def a_negative(r, beta, h, s_l):
    # + beta - sqrt (unstable solution)
    return 1/16*(2*np.pi*r - np.sqrt(np.pi) + beta - h - np.sqrt((2*np.pi*r - np.sqrt(np.pi) + beta - h)**2 - 32 * s_l))


def a_stable(r, beta, h, s_l):
    # - beta no sqrt
    return 1/16*(2*np.pi*r - np.sqrt(np.pi) + beta - h)


def minimun_leader_strength(r, beta, h):
    return (2*np.pi*r - np.sqrt(np.pi) - h)/beta


def maximun_leader_strength(r, beta, h):
    # TODO: Add with -beta, but one should be enough
    return (1/32)*(2*np.pi*r - np.sqrt(np.pi) + beta - h)**2




def find_critical_temperature(tmin,tmax,timesteps,t_values,sims_per_timestep,threshold,GRIDSIZE_X,GRIDSIZE_Y,BETA_PEOPLE,BETA_LEADER,H,P_OCCUPATION,P_OPINION_1,S_LEADER,a_0,S_MEAN):

    temperatures = np.linspace(tmin,tmax,t_values)
    p_overcoming_leader = np.zeros(t_values)

    for T in (range(t_values)):
        
        # Retrieve the T
        TEMP = temperatures[T]
        leader_overcomed = 0
        print(f'Sim {T+1}/{t_values}: {TEMP}')
        
        with Pool() as pool:
            # Do many simulations with that T to get av. cluster size
            # Each with a different innitialization
            for sim in range(sims_per_timestep):
                # Update temperature of model!

                model = CA(gridsize_x=GRIDSIZE_X, gridsize_y=GRIDSIZE_Y, temp=TEMP, beta=BETA_PEOPLE, beta_leader=BETA_LEADER, h=H, p_occupation=P_OCCUPATION, p_opinion_1=P_OPINION_1, s_leader=S_LEADER, s_mean=S_MEAN)

                #pool.map(simulate_single_run, [sim]*sims_per_timestep)
                #TODO: Evolve one ts at a time so if we overcome leader, 
                # we can end this sim already

                data = model.evolve(timesteps)
                simulation = data['opinions']
                last_cluster_size = data['cluster_sizes'][-1]

                if last_cluster_size <= threshold:
                    leader_overcomed += 1

        # Get average cluster size
    p_overcoming_leader[T] = leader_overcomed / sims_per_timestep


    return temperatures,p_overcoming_leader



def plot_critical_temperature(temperatures,average_cluster_sizes,xmin,xmax,T_VALUES,R):

    fig,ax = plt.subplots()

    #plt.figure()

    # Max cluster size line
    x_cons = np.linspace(xmin, xmax, T_VALUES)
    y_cons = np.ones(T_VALUES)*R
    ax.plot(x_cons,y_cons,c='black',linestyle='-')

    # Horizontal line with minimun value expected cluster

    # Add threshold used to figure caption!

    ax.plot(temperatures,average_cluster_sizes)
    ax.set_title('Probability of overcomming leader cluster with temperature')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('p(Overcoming leader cluster)')
    ax.set_xlim([temperatures[0],temperatures[-1]])
    ax.set_ylim([0,1])

    #plt.grid()
    #plt.tight_layout()
    #plt.show()

    return fig,ax