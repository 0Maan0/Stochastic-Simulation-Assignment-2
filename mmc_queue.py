'''
This module contains the main implementation of an mmn queue simulation.
'''
import numpy as np
import matplotlib.pyplot as plt
import simpy as sp

def system_load(lam, mu, n):
    '''
    Calculate the system load of an M/M/n queue.
    return: float
    '''
    rho = lam / (n*mu)
    return rho

def service(env, name, server, waiting_times, service_time):
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
    for i in range(max_customers):
        inter_arrival = np.random.exponential(1/lam)
        yield env.timeout(inter_arrival)
        service_time = np.random.exponential(1/mu)
        env.process(service(env, f'Customer {i}', server, waiting_times, service_time))

def simulate_MMn_queue(lam, mu, n, max_customers = 10000, seed=None):
    if seed:
        np.random.seed(seed)
    rho = system_load(lam, mu, n)
    if rho >= 1:
        ValueError("System is unstable")
    env = sp.Environment()
    server = sp.Resource(env, capacity=n)
    waiting_times = []
    env.process(source(env, lam, server, mu, waiting_times, max_customers))
    env.run()
    return waiting_times

lambd = 1
mu = 1
n_tests = [1, 2, 4]
for n in n_tests:
    simulation = simulate_MMn_queue(lambd, mu, n, seed=0)
    print(f"for n = {n} the mean waiting time is {np.mean(simulation)} and the std is {np.std(simulation)}.")	
'''
def steady_state_probability(rho, n):
    part1 = np.sum((n*rho)**i / np.math.factorial(i) for i in range(n))
    part2 = ((n*rho)) **n / np.math.factorial(n) * (1/(1 - rho))
    p0 = 1 / (part1 + part2)
    return p0

        # Mean number of custormers in the system
    N = n*rho + rho*(n*rho)**n / (np.math.factorial(n)) *p0/(1-rho)
'''