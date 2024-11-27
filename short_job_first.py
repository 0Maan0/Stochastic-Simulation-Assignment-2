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

        self.queue_priority = [] # set of tuples: (p, service_time)
        self.queue_time = [] # set of tuples: (p, service_time)

        if self.rho >= 1:
            raise ValueError("System is unstable")

    def service(self, service_time, priority):
        '''
        The traject of a customer in the system. (Arrival -> Service -> Departure).
        '''
        arrive = self.env.now

        print('-------------------------')
        print(self.queue_priority)
        print(self.queue_time)
        print('-------------------------')
        print()

        with self.server.request(priority=priority) as req:
            yield req # Wait until the request is granted.
            wait = self.env.now - arrive

            # update queue of priorities, so that new customers get a relevant priority
            # takeout shortest job. since it sjf
            index = np.array(self.queue_time).argmin()
            print(service_time, self.queue_time[index] )
            self.queue_time.pop(index)
            self.queue_priority.pop(index)
            

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
            priority = int(service_time*1000)

            self.queue_priority.append(priority)
            self.queue_time.append(service_time)

            self.env.process(self.service(service_time, priority))

    def simulate_mmn_queue(self, max_customers = 200, seed=None):
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
MU = 0.6
n = 2


sim = sims(LAMBDA, MU, n)
rho, simulation = sim.simulate_mmn_queue()
print(f"for mu = {MU}, n = {n} and {rho=} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}.")