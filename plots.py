import numpy as np
from mmn_queue import sims 
import matplotlib.pyplot as plt

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