'''
This module contains the main implementation of an M/M/N queue simulation.
'''
import numpy as np
import matplotlib.pyplot as plt
import simpy
class sims():

    def __init__(self, lam, mu, n):
        '''
        Calculate the system load of an M/M/n queue.
        return: float
        '''
        self.rho = lam / (n*mu)
        self.lam = lam
        self.n = n
        self.mu = mu

        self.queue = {} # set of tuples: (p, service_time)

        if self.rho >= 1:
            raise ValueError("System is unstable")

    def service(self, id, service_time, priority):
        '''
        The traject of a customer in the system. (Arrival -> Service -> Departure).
        '''
        arrive = self.env.now
        with self.server.request(priority=priority) as req:
            yield req # Wait until the request is granted.
            wait = self.env.now - arrive
            self.queue.remove((priority, service_time))
            self.waiting_times.append(wait)
            yield self.env.timeout(service_time) # Wait for the service to finish.


    
    def source(self, max_customers):
        '''
        Generate customers at rate lambda.
        '''
        for _ in range(max_customers):
            inter_arrival = np.random.exponential(1/self.lam) # Time between arrivals.
            yield self.env.timeout(inter_arrival)
            service_time = np.random.exponential(1/self.mu)
            # find right p: priority
            priority = max_customers
            for queue_priority, queue_time in self.queue:
                if service_time < queue_time:
                    priority = queue_priority - 1
            self.queue.append((priority, service_time))
            id = len(self.queue)
            #print(self.queue)
            self.env.process(self.service(id, service_time, priority))

    def simulate_mmn_queue(self, max_customers = 10000, seed=None):
        '''
        Start environment and run the simulation.
        return: list
        '''
        if seed:
            np.random.seed(seed)

        self.env = simpy.Environment()
        self.server = simpy.PriorityResource(self.env, capacity=n)
        self.waiting_times = []
        self.env.process(self.source(max_customers))
        self.env.run()
        return self.rho, self.waiting_times

LAMBDA = 1
mu_values = np.linspace(1.1, 4, 10)
n_tests = [1, 2, 4]


for MU in mu_values:
    for n in n_tests:
        sim = sims(LAMBDA, MU, n)
        rho, simulation = sim.simulate_mmn_queue()
        print(f"for mu = {MU}, n = {n} and {rho=} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}.")
    quit()

MU = 1.1
mean_waiting_times = []
std_waiting_times = []
precision = 0.01
std = 1
mean = 0
while std > precision* mean:
    simulation = simulate_mmn_queue(LAMBDA, MU, 2) 
    mean_waiting_times.append(np.mean(simulation))
    std_waiting_times.append(np.std(simulation))

    mean = np.mean(mean_waiting_times)
    std = np.std(mean_waiting_times)
print(f"for mu = {MU} and n = {2} the mean waiting time is {mean} +- {std}.")

