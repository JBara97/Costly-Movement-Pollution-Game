import numpy as np
import scipy.ndimage as ndimage
import random
import math
import networkx as nx

## Pollution with COSTLY MIGRATION

def points_within_range(position,radius,size=1):
    """points within range of position at a distance <= radius,
        on the double-torus of period size x size
            position - (int,int) tuple
            radius - float
            size - int
            """
    return [(x%size,y%size) for x in range(position[0]-radius,
                                    position[0]+radius+1)
            for y in range(position[1]-radius,
                            position[1]+radius+1)
            if np.sqrt((x-position[0])**2+(y-position[1])**2) <= radius]

def torus_distance(p0,p1,L):
    d = np.abs(p1-p0)
    return np.linalg.norm(np.array([d, L-d]).min(axis=0),axis=0)

class Agent:
    """
    Agent object with the following methods:
        self.pollute() : add/remove pollution from the world pollution grid
        self.observe() : return pollution at self.position
        self.migrate() : migrating to a a location with least
                            amount of pollution and cost to move within radius
                            self.migration
        self.imitate() : imitating the strategy of an agent who's site
                            experiences the lowest expense,
                            within radius self.imitate_radius
        self.calc_expense() : calculates and sets self.expense
    """
    def __init__(self,position=(0,0),type='c',R = 5,M_nu=1,
                    label=1,phi=5,epsilon=0,mu=0.5):
        self.label = label # must be a positive number > 0
        self.position = position # position = (x,y) as a tuple of integers
        self.type = type.lower() # type = 'c' or 'd'
        self.imitate_radius = M_nu # imitation radius, if M_nu<1 then no imitation!
        self.phi = phi # cleaning rate of a cooperator
        self.expense = 0 # initially 0 expense
        self.mutation = epsilon # float in [0,1]
        self.radius=R # float
        self.dist_moved = 0 # float
        self.mu = mu

    # def pollute(self,world):
    #     """ self.pollute() : add/remove pollution from the world pollution grid """
    #     if self.type=='c':
    #         affected_pts = points_within_range(self.position,1,world.size)
    #         for pt in affected_pts:
    #             world.pollution_grid[pt] -= self.phi
    #     elif self.type=='d':
    #         affected_pts = points_within_range(self.position,self.radius,world.size)
    #         for pt in affected_pts:
    #             x,y=pt
    #             r = np.sqrt((x-self.position[0])**2+(y-self.position[1])**2)
    #             if r <= 1:
    #                 world.pollution_grid[pt] += 1
    #             elif r <= self.radius:
    #                 world.pollution_grid[pt] += 1/r**2
    #     else:
    #         print("self.type error, must be c or d. Will pollute 0 everywhere")

    def pollute(self,world):
        """ self.pollute() : add/remove pollution from the world pollution grid """
        if self.type=='c':
            world.pollution_grid[torus_distance(np.array(self.position).reshape(2,1,1),
                                    world.position_grid,world.size)<=1] -= self.phi
        elif self.type=='d':
            dis_grid = torus_distance(np.array(self.position).reshape(2,1,1),
                                                world.position_grid,world.size)
            world.pollution_grid[dis_grid<1]+=1
            mask = (dis_grid>=1)&(dis_grid<self.radius)
            world.pollution_grid[mask]+=1/dis_grid[mask]**2
        else:
            print("self.type error, must be c or d. Will pollute 0 everywhere")

    def observe(self,world):
        """ self.observe() : return pollution at self.position """
        return world.pollution_grid[self.position]

    def migrate(self,world):
        """self.migrate() : migrating to a a location with least
                            amount of pollution within radius
                            self.migration"""
        pol = self.observe(world)
        cost_grid = world.pollution_grid + self.mu*torus_distance(np.array(self.position).reshape((2,1,1)),
                                                    world.position_grid,world.size)
        min_pol = cost_grid[world.lattice_sites==0].min()
        min_positions = np.where(cost_grid==min_pol)
        if (pol > min_pol)&(np.size(min_positions)>0):
            # restricting candidate_pts to those with the smallest pollution
            min_positions = list(zip(min_positions[0],min_positions[1]))
            new_pos = random.choice(min_positions)
            world.lattice_sites[self.position] = 0 # emptying world site
            self.dist_moved = torus_distance(np.array(self.position),np.array(new_pos),world.size)
            self.position = new_pos # for multiple minima
            world.lattice_sites[self.position] = self.label # moving to another world site
        else:
            self.dist_moved = 0

    def imitate(self,world):
        """imitating the strategy of an agent who's site
            experiences the lowest expense,
            within radius self.imitate_radius"""

        # Mutation with probability epsilon to flip strategies
        if np.random.uniform()<self.mutation:
            if self.type=='c':
                self.type='d'
            else:
                self.type='c'

        # Imitate the neighbour with minimum expense
        elif self.imitate_radius>=1:
            neighbours = [world.return_agent(world.lattice_sites[pt])[0] for pt in
                                points_within_range(self.position,
                                            self.imitate_radius,
                                            world.size)
                                if world.lattice_sites[pt]>0]
            neighbour_expense = {a:a.expense for a in neighbours}
            best_neighbours = [k for k, v in neighbour_expense.items() if v==min(neighbour_expense.values())]
            if len(best_neighbours)>0:
                self.type = np.random.choice(best_neighbours).type

    def calc_expense(self,world):
        self.expense = self.observe(world) + self.mu*self.dist_moved
        if self.type == 'c':
            self.expense += world.f
        elif self.type =='d':
            self.expense -= world.g


class World:
    """
    World object with the following methods:
        self.populate() : fill self.lattice_sites with agents with parameters
                            (either N with D defectors, or from the list agents)
        self.step() : progress the world by one timestep -
                        1. All agents update strategies
                        2. All agents migrate
                        (3.) self.pollution_grid is reset
                        4. All agents pollute
                        5. All agents calculate expense
        self.pollute() : all agents pollute
        self.calc_expense() : all agents calculate calculate expense
        self.migrate() : all agents migrate
        self.imitate() : all agents update their strategies
        self.spatial_avg() : return spatial average of pollution
                                (ie mean over all lattice sites)
        self.per_capita_pollution() : return per-capita POLLUTION
                                        (ie mean over all occupied sites)
        self.cleaner_rate() : return fraction of cooperators (C/N) in the city
        self.per_capita_expense() : return per-capita EXPENSE
        self.observe_clusters() : returns the list of clusters,
                                    via NetworkX connected_components
        self.return_agent(label) : return the agent object that matches the label
    """
    def __init__(self,L=50,N=10,D=1,agents=[],R=5,phi=5,M_nu=1,
                    f=1,g=2,epsilon=0,mu=0.5):
        self.size = L
        self.pollution_grid = np.zeros([L,L],dtype=np.float64) # the pollution grid / space
        self.lattice_sites = np.zeros([L,L]) # 0 or a label
        self.f=f
        self.g=g
        self.mutation = epsilon
        self.position_grid = np.transpose(np.array(np.meshgrid(np.arange(L),np.arange(L))),axes=(0,2,1))
        self.populate(N=N,D=D,agents=agents,R=R,phi=phi,M_nu=M_nu,mu=mu)
        self.pollute()
        self.calc_expense()

    def populate(self,N=10,D=1,agents=[],R=5,phi=5,M_nu=1,epsilon=0,mu=0.5):
        """ self.populate() : fill self.lattice_sites with agents
                agents is a list of Agent objects, which will supersed N and D
        """
        if len(agents) > 0:
            for a in agents:
                self.lattice_sites[a.position] = a.label
            self.agents = agents
        else:
            empty_sites = [tuple(item) for item in np.argwhere(self.lattice_sites==0).tolist()]
            to_be_inhabited = random.sample(empty_sites,N)
            agents = [Agent(position=to_be_inhabited[i],
                            type='d',label=i+1,R=R,phi=phi,M_nu=M_nu,epsilon=epsilon)
                            for i in range(D)]
            agents += [Agent(position=to_be_inhabited[i],
                                type='c',label=i+1,R=R,phi=phi,M_nu=M_nu,epsilon=epsilon)
                                for i in range(D,N)]
            if type(mu) in [float,int,np.float64]:
                for a in agents:
                    a.mu = mu
            else:
                for i in range(len(agents)):
                    agents[i].mu = mu[i]
            for a in agents:
                self.lattice_sites[a.position] = a.label
            random.shuffle(agents)
            self.agents= agents

    def step(self):
        """
        self.step() : progress the world by one timestep -
                        1. All agents update strategies
                        2. All agents migrate
                        3. All agents pollute
                        4. All agents calculate expense
        """
        self.imitate()
        self.migrate()
        self.pollute()
        self.calc_expense()

    def pollute(self):
        """ self.pollute() : all agents pollute """
        self.pollution_grid=np.zeros([self.size,self.size]) # resets every time step
        for a in self.agents:
            a.pollute(self)

    def calc_expense(self):
        """ self.calc_expense() : all agents calculate calculate expense """
        for a in self.agents:
            a.calc_expense(self)

    def migrate(self):
        """ self.migrate() : all agents migrate """
        for a in self.agents:
            a.migrate(self)

    def imitate(self):
        """ self.imitate() : all agents update strategies"""
        for a in self.agents:
            a.imitate(self)

    def spatial_avg(self):
        """ self.spatial_avg() : return spatial average of pollution
                                (ie mean over all lattice sites) """
        return np.mean(self.pollution_grid)

    def per_capita_pollution(self):
        """ self.per_capita_pollution() : return per-capita POLLUTION """
        return self.pollution_grid[self.lattice_sites!=0].mean()

    def cleaner_rate(self):
        """ self.cleaner_rate() : return fraction of cooperators (C/N) in the city """
        return len([a for a in self.agents if a.type=='c'])/len(self.agents)

    def per_capita_expense(self):
        """ self.per_capita_expense() : return per-capita EXPENSE """
        return np.mean([a.expense for a in self.agents])

    def observe_clusters(self,output='labels'):
        """ self.observe_clusters() : returns the list of clusters,
                                        via scipy.ndimage.label """
        # Convert the grid into a labelled array
        label_image = ndimage.label(self.lattice_sites)[0]
        # Impose periodic boundaries
        for y in range(label_image.shape[0]):
            if label_image[y, 0] > 0 and label_image[y, -1] > 0:
                label_image[label_image == label_image[y, -1]] = label_image[y, 0]
        for x in range(label_image.shape[1]):
            if label_image[0, x] > 0 and label_image[-1, x] > 0:
                label_image[label_image == label_image[-1, x]] = label_image[0, x]
        if output.lower() == 'labels':
            return [set(a.label for a in self.agents
                        if label_image[a.position]==val)
                    for val in set(label_image[label_image>0])
                    if len(label_image[label_image==val]) > 1]
        elif output.lower() == 'agents':
            return [set(a for a in self.agents
                        if label_image[a.position]==val)
                    for val in set(label_image[label_image>0])
                    if len(label_image[label_image==val]) > 1]

    def return_agent(self,label):
        """ self.return_agent(label) : return the agent object that matches the label """
        return [a for a in self.agents if a.label==label]

    def cluster_breakdown(self,by_strat=False,ccs=None,mu_vals=None):
        """ self.cluster_breakdown() : return a (len(mu_list), 2) ndarray of fraction of agents """
        if ccs is None:
            ccs = self.observe_clusters()
        ccs_sets = ccs[0]
        for i in range(1,len(ccs)):
            ccs_sets = ccs_sets.union(ccs[i])

        if by_strat:
            breakdown = np.array([np.nan,np.nan])
            c = [a for a in self.agents if a.type=='c']
            d = [a for a in self.agents if a.type=='d']
            if len(c)>0:
                breakdown[0] = len([a for a in c if a.label in ccs_sets])/len(c)
            if len(d)>0:
                breakdown[1] = len([a for a in d if a.label in ccs_sets])/len(d)
        else:
            if mu_vals is None:
                mu_vals = set(a.mu for a in self.agents)
            breakdown = np.zeros([len(mu_vals),2]) # wealth x c/d
            for i,mu in enumerate(mu_vals):
                c = [a for a in self.agents if a.type=='c' and a.mu==mu]
                d = [a for a in self.agents if a.type=='d' and a.mu==mu]
                if len(c)>0:
                    breakdown[i,0] = len([a for a in c if a.label in ccs_sets])/len(c)
                else:
                    breakdown[i,0] = np.nan
                if len(d)>0:
                    breakdown[i,1] = len([a for a in d if a.label in ccs_sets])/len(d)
                else:
                    breakdown[i,1] = np.nan
        return breakdown # will return a (#mu,2) ndarray

    def get_type_grid(self,type_map={'c':1,'d':-1}):
        grid = np.zeros_like(self.pollution_grid)
        for a in self.agents:
            grid[a.position] = type_map[a.type]
        return grid
