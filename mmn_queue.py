"""
This module simulates and analyzes queuing systems using the SimPy library. 
It includes the 'sims' class, which allows for the modeling of various queue configurations and 
scheduling policies.

Key Features:
- Models M/M/n and M/D/1 queuing systems.
- Supports First-In-First-Out (FIFO) and Shortest Job First (SJF) scheduling.
- Includes options for deterministic service times and long-tail service time distributions.
- Simulates customer behavior from arrival through service completion.

Classes:
- sims: Represents a queueing system with configurable parameters for arrival rate, service rate, 
  number of servers, scheduling method, and service time distribution.

Dependencies:
- numpy: Random number generation and Numerical computations.
- simpy: Event-driven simulation of queuing system.

Raise:
- Raises 'ValueError' if the system load (ρ) is unstable (i.e., ρ >= 1).
"""

import numpy as np
import matplotlib.pyplot as plt
import simpy


class sims():
    """ 
    Simulates an an M/M/n or M/M/1 queuing system with either a short-job-first or FIFO scheduling system.

    This class models the arrival, service, and departure of customers in a queuing system using 
    the SimPy library. It can handle various configurations of queuing systems, including:
    - Multi-server queues (M/M/n)
    - Single-server queues with deterministic service times (M/D/1)
    - Queues with Shortest Job First (SJF) scheduling
    - Queues with long-tail service time distributions.
    """


    def __init__(self, lam, mu, n, rho=None, sjf=False, deterministic=False, tail=False):
        '''
        Calculate the system load of an M/M/n queue. or M/D/1 queue if deterministic
        
        Parameters:
        lam (float): Arrival rate (λ) of customers into the system.
        mu (float): Service rate (μ) of servers.
        n (int): Number of servers in the system.
        rho (float, optional): System load factor (ρ). If provided, it overrides `mu`.
        sjf (bool, optional): Default false, determines if Shortest Job First (SJF) scheduling is enabled.
        deterministic (bool, optional): Default false, determines if deterministic service times (M/D/1) is used.
        tail (bool, optional): default False, determines if a long-tail (hyperexponential) distribution for service times is used.

        Raises:
            ValueError: If the system load factor (ρ) is greater than or equal to 1, indicating an unstable system.
        '''

        self.lam = lam
        self.n = n

        if rho is not None:
            #Setting rho ignores the given mu.
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
        Models the service process for a single customer, including their waiting time and service duration.
        '''

        arrive = self.env.now

        with self.server.request(priority=priority) as req:
            yield req #Wait until the request is granted.
            wait = self.env.now - arrive

            self.waiting_times.append(wait)
            self.service_time_list.append(service_time)

            yield self.env.timeout(service_time) #Wait for the service to finish.

    
    def source(self, max_customers):
        '''
        Generates customer arrivals according to the specified arrival rate (λ).
        '''

        for _ in range(max_customers):
            #Determine the time between arrivals.
            inter_arrival = np.random.exponential(1/self.lam) 
            yield self.env.timeout(inter_arrival)

            #Determintistic distribution. 1/mu M/D/n.
            if self.deterministic:
                service_time = 1 / self.mu

            #Long-tail distribution, hyperexponential. 
            elif self.tail:
                if np.random.rand() <= 0.25:
                    service_time = np.random.exponential(5)
                else:
                    service_time = np.random.exponential(1)
            
            #Exponential distribution, M/M/n.
            else:
                service_time = np.random.exponential(1/self.mu)

            #Set the priority based on service time value.
            if self.sjf:
                priority = int(service_time*1000)
            else:
                priority = 0

            self.env.process(self.service(service_time, priority))


    def simulate_queue(self, max_customers = 10000, seed=None):
        '''
        Simulates the queueing system and returns a list of waiting times for all customers.
        Return: list of waiting times.
        '''

        #Lists used for tracking the simulation.
        self.waiting_times = [] #Time waited, result of sim.
        self.service_time_list = [] #Follows M distributions.

        if seed:
            np.random.seed(seed)

        self.env = simpy.Environment()
        self.server = simpy.PriorityResource(self.env, capacity=self.n)

        self.env.process(self.source(max_customers))
        self.env.run()

        return self.waiting_times

"""

LAMBDA = 1
MU = 1.1
n = 2

if False:
    sim = sims(LAMBDA, MU, n, rho=0.9, tail=True)
    rho, simulation = sim.simulate_queue()
    print(f"for mu = {sim.mu}, n = {sim.n} and rho = {sim.rho} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}, tail = {sim.tail}, sjf = {sim.sjf}, deterministic = {sim.deterministic}.")
sim = sims(LAMBDA, MU, n, rho=0.9, sjf=True)
rho, simulation = sim.simulate_queue()
print(f"for mu = {sim.mu}, n = {sim.n} and rho = {sim.rho} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}, sjf = {sim.sjf}, deterministic = {sim.deterministic}.")

sim2 = sims(LAMBDA, MU, n, rho=0.9, sjf=False)
_, _ = sim2.simulate_queue()

print(f"for mu = {sim2.mu}, n = {sim2.n} and rho = {sim2.rho} the mean waiting time is {np.mean(sim2.waiting_times)} +- {np.std(sim2.waiting_times)}, sjf = {sim2.sjf}., deterministic = {sim2.deterministic}.")

print('-----')
print(np.mean(sim.service_time_list))
print(np.mean(sim2.service_time_list))

fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, gridspec_kw={'wspace': 0.05})

ax[0].scatter(sim.service_time_list, simulation, marker='.', label='SJF', s=50, lw=0)
ax[1].scatter(sim2.service_time_list, sim2.waiting_times, marker='.', label='FIFO', s=50, lw=0)
ax[0].set_ylabel(r'Waiting Time ($t_w$)')
ax[0].set_xlabel(r'Service Time ($t_s$)')
ax[1].set_xlabel(r'Service Time ($t_s$)')
ax[0].legend()
ax[1].legend()

plt.show()
plt.savefig(
      f'Figures/sjf_vs_fifo_n{n}_rho{rho}.pdf',
      bbox_inches='tight', format='pdf'
)

plt.show()

"""