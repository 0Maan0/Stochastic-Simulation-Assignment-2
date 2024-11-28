'''
This module contains the main implementation of an M/M/N queue simulation.
'''
import numpy as np
import matplotlib.pyplot as plt
import simpy



class sims():

    def __init__(self, lam, mu, n, rho=None, sjf=False, deterministic=False, tail=False):
        '''
        Calculate the system load of an M/M/n queue. or M/D/1 queue if deterministic
        return: float
        '''

        self.lam = lam
        self.n = n

        if rho is not None:
            # setting rho ignores given mu
            self.rho = rho
            self.mu = lam/n/rho
        else:
            self.rho = lam / (n*mu)
            self.mu = mu

        if self.rho >= 1:
            raise ValueError("System is unstable")
        
        self.sjf = sjf
        self.deterministic = deterministic
        self.tail = tail

    def service(self, service_time, priority):
        '''
        The traject of a customer in the system. (Arrival -> Service -> Departure).
        '''
        arrive = self.env.now
        with self.server.request(priority=priority) as req:
            yield req # Wait until the request is granted.
            wait = self.env.now - arrive

            self.waiting_times.append(wait)
            self.service_time_list.append(service_time)

            yield self.env.timeout(service_time) # Wait for the service to finish.


    
    def source(self, max_customers):
        '''
        Generate customers at rate lambda.
        '''
        for _ in range(max_customers):
            inter_arrival = np.random.exponential(1/self.lam) # Time between arrivals.
            yield self.env.timeout(inter_arrival)


            if self.deterministic:
                # determintistic dist. 1/mu m/d/n
                service_time = 1 / self.mu

            elif self.tail:
                # long-tail dist. hyperexponential. 
                if np.random.rand() <= 0.25:
                    service_time = np.random.exponential(5)
                else:
                    service_time = np.random.exponential(1)
            else:
                # exponential dist. m/m/n
                service_time = np.random.exponential(1/self.mu)

            if self.sjf:
                #  set priority based on service time value
                priority = int(service_time*1000)
            else:
                priority = 0

            self.env.process(self.service(service_time, priority))

    def simulate_queue(self, max_customers = 10000, seed=None):
        '''
        Start environment and run the simulation.
        return: list
        '''
        # lists used for tracking sim
        self.waiting_times = [] # time waited, result of sim
        self.service_time_list = [] # follows M dist.

        if seed:
            np.random.seed(seed)

        self.env = simpy.Environment()
        self.server = simpy.PriorityResource(self.env, capacity=self.n)


        self.env.process(self.source(max_customers))
        self.env.run()
        return self.waiting_times

LAMBDA = 1
MU = 1.1
n = 2

if False:
    sim = sims(LAMBDA, MU, n, rho=0.9, tail=True)
    rho, simulation = sim.simulate_queue()
    print(f"for mu = {sim.mu}, n = {sim.n} and rho = {sim.rho} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}, tail = {sim.tail}, sjf = {sim.sjf}, deterministic = {sim.deterministic}.")

    sim2 = sims(LAMBDA, MU, n, rho=0.9, tail=False)
    _, _ = sim2.simulate_queue()
    print(f"for mu = {sim2.mu}, n = {sim2.n} and rho = {sim2.rho} the mean waiting time is {np.mean(sim2.waiting_times)} +- {np.std(sim2.waiting_times)}, tail = {sim2.tail}, sjf = {sim2.sjf}., deterministic = {sim2.deterministic}.")

    print('-----')
    print(np.mean(sim.service_time_list))
    print(np.mean(sim2.service_time_list))

    fig, ax = plt.subplots(1, 2, sharey=True)

    ax[0].scatter(sim.service_time_list, simulation, marker='.', label='tail')
    ax[1].scatter(sim2.service_time_list, sim2.waiting_times, marker='.', label='M')
    ax[0].set_ylabel('waiting time')
    ax[0].set_xlabel('service time')
    ax[1].set_ylabel('waiting time')
    ax[1].set_xlabel('service time')
    ax[0].legend()
    ax[1].legend()

    plt.show()
LAMBDA = 1
MU = 1.2
n_values = [4, 2, 1]
rho_values = [0.8, 0.85, 0.9, 0.95, 0.99]

mean_waiting_times = []
for rho in rho_values:
    plt.figure()
    for n in n_values:
        sim = sims(LAMBDA, MU, n, rho=rho)
        simulation = sim.simulate_queue(max_customers = 1000)
        plt.hist(simulation, bins=30, alpha=0.5, label=f'n = {n}')
        plt.legend()
    plt.xlabel('waiting time')
    plt.ylabel('frequency')
    plt.title(f'rho = {rho}')
    plt.show()
"""
sim = sims(LAMBDA, MU, n, rho=0.9)
print(np.linspace(11, 200, num=90))
mean_waiting_times = []
std_waiting_times = []
mean_values = np.zeros(190)
std_values = np.zeros(190)
for i in range(200):
    simulation = sim.simulate_queue(max_customers = 10000)
    mean_waiting_times.append(np.mean(simulation))
    std_waiting_times.append(np.std(simulation, ddof=1))
    if len(mean_waiting_times) < 11:
        continue
    mean_values[i-11] = np.mean(mean_waiting_times)
    std_values[i-11] = np.std(mean_waiting_times, ddof=1)
    print(i)
plt.figure()
plt.errorbar(np.linspace(11, 200, num=190), mean_values, yerr=std_values, fmt='o')
plt.ylabel('mean waiting time')
plt.xlabel('number of simulations')
plt.show()
"""