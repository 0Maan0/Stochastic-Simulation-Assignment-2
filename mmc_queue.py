'''
This module contains the main implementation of an M/M/N queue simulation.
'''
import numpy as np
import matplotlib.pyplot as plt
import simpy

def system_load(lam, mu, n):
    '''
    Calculate the system load of an M/M/n queue.
    return: float
    '''
    rho = lam / (n*mu)
    return rho

def service(env, server, waiting_times, service_time):
    '''
    The traject of a customer in the system. (Arrival -> Service -> Departure).
    '''
    arrive = env.now
    with server.request() as req:
        yield req # Wait until the request is granted.
        wait = env.now - arrive
        waiting_times.append(wait)
        yield env.timeout(service_time) # Wait for the service to finish.
 
def source(env, lam, server, mu, waiting_times, max_customers):
    '''
    Generate customers at rate lambda.
    '''
    for _ in range(max_customers):
        inter_arrival = np.random.exponential(1/lam) # Time between arrivals.
        yield env.timeout(inter_arrival)
        service_time = np.random.exponential(1/mu)
        env.process(service(env, server, waiting_times, service_time))

def simulate_mmn_queue(lam, mu, n, max_customers = 10000, seed=None):
    '''
    Start environment and run the simulation.
    return: list
    '''
    if seed:
        np.random.seed(seed)
    rho = system_load(lam, mu, n)
    if rho >= 1:
        raise ValueError("System is unstable")
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=n)
    waiting_times = []
    env.process(source(env, lam, server, mu, waiting_times, max_customers))
    env.run()
    return waiting_times

LAMBDA = 1
mu_values = np.linspace(1.1, 4, 2)
n_tests = [1, 2, 4]
for MU in mu_values:
    for N in n_tests:
        simulation = simulate_mmn_queue(LAMBDA, MU, N)
        print(f"for mu = {MU} and n = {N} the mean waiting time is {np.mean(simulation)} +- {np.std(simulation)}.")


MU = 1.2
mean_waiting_times = []
std_waiting_times = []
precision = 0.05
std = 1
mean = 0
n = 0
while std > precision * mean:
    print(precision * mean)
    print(n)
    simulations = [simulate_mmn_queue(LAMBDA, MU, 2, max_customers = 10000) for _ in range(2)]
    for simulation in simulations:
        mean_waiting_times.append(np.mean(simulation))
        std_waiting_times.append(np.std(simulation, ddof=1))
    n += 2
    if n < 20:
        continue
    mean = np.mean(mean_waiting_times)
    print(len(mean_waiting_times))
    std = np.std(mean_waiting_times, ddof=1)
    #print(f"mean: {mean}")
    print(f"standard deviation: {std}")
print(f"for mu = {MU} and n = {2} the mean waiting time is {mean} +- {std}.")
