'''
This module contains the main implementation of an M/M/N queue simulation.
'''
import numpy as np
import matplotlib.pyplot as plt
import simpy



class sims():

    def __init__(self, lam, mu, n, rho=None, sjf=False):
        '''
        Calculate the system load of an M/M/n queue.
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


            service_time = np.random.exponential(1/self.mu)

            if self.sjf:
                #  set priority based on service time value
                priority = int(service_time*1000)
            else:
                priority = 0

            self.env.process(self.service(service_time, priority))

    def simulate_mmn_queue(self, max_customers = 1000, seed=None):
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
        self.server = simpy.PriorityResource(self.env, capacity=n)


        self.env.process(self.source(max_customers))
        self.env.run()
        return self.rho, self.waiting_times

LAMBDA = 1
MU = 0.6
n = 2


sim = sims(LAMBDA, MU, n, rho=0.9, sjf=True)
rho, simulation = sim.simulate_mmn_queue()
print(f"for mu = {sim.mu}, n = {sim.n} and rho = {sim.rho} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}, sjf = {sim.sjf}.")

sim2 = sims(LAMBDA, MU, n, rho=0.9, sjf=False)
_, _ = sim2.simulate_mmn_queue()
print(f"for mu = {sim2.mu}, n = {sim2.n} and rho = {sim2.rho} the mean waiting time is {np.mean(sim2.waiting_times)} +- {np.std(sim2.waiting_times)}, sjf = {sim2.sjf}.")

fig, ax = plt.subplots(1, 2, sharey=True)

ax[0].scatter(sim.service_time_list, simulation, marker='.', label='sfj')
ax[1].scatter(sim2.service_time_list, sim2.waiting_times, marker='.', label='fifo')
ax[0].set_ylabel('waiting time')
ax[0].set_xlabel('service time')
ax[1].set_ylabel('waiting time')
ax[1].set_xlabel('service time')
ax[0].legend()
ax[1].legend()

plt.show()