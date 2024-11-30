'''
This module contains code to plot simulated queues from the mmn_queue.py module.
'''
import numpy as np
import matplotlib.pyplot as plt
from mmn_queue import sims

def plot_mean_waiting_times(lambd, mu, n, rho_values, num_simulations, max_customers, starting_amount):
    '''
    Plot the mean of the mean waiting times for a certain number of simulations. 
    '''
    FONTSIZE = 16
    fig, axes = plt.subplots(1, len(rho_values), figsize=(len(rho_values)*5, 5), sharey=True)

    for j, rho in enumerate(rho_values):
        ax = axes[j]
        sim = sims(lambd, mu, n, rho=rho)

        mean_waiting_times = []
        mean_values = []
        std_values = []

        for _ in range(num_simulations):
            simulation = sim.simulate_queue(max_customers=max_customers)
            mean_waiting_times.append(np.mean(simulation))  # Add newest mean waiting time to list.

            if len(mean_waiting_times) >= starting_amount: # Start calculating mean and std after starting_amount simulations.
                current_mean = np.mean(mean_waiting_times) # Calculate mean and std of mean waiting times.
                current_std = np.std(mean_waiting_times, ddof=1)
                mean_values.append(current_mean)
                std_values.append(current_std)

        x_values = np.linspace(starting_amount, num_simulations, num=len(mean_values)) # x array for number of simulations.
        mean_values = np.array(mean_values)
        std_values = np.array(std_values)

        ax.fill_between(x_values, mean_values - std_values, mean_values + std_values, color='tab:blue', alpha=0.3, label=r'Standard Deviation $\sigma$')
        ax.plot(x_values, mean_values, color='tab:blue', label='Mean Waiting Time')
        ax.axhline(y=np.mean(mean_waiting_times), color='tab:red', linestyle='--', label='Average Mean Waiting Time')
        
        if ax == axes[0]:
            ax.set_ylabel('Mean Waiting Time', fontsize=FONTSIZE)
            ax.legend(loc = 'upper left', fontsize=FONTSIZE-4)
        ax.set_xlabel('Number of Simulations', fontsize=FONTSIZE)
    
        ax.text(0.95, 0.95, r'$\rho$ =' + f'{rho}', transform=ax.transAxes, fontsize=FONTSIZE,
        verticalalignment='top', horizontalalignment='right')
        ax.tick_params(axis='x', labelsize=FONTSIZE-2)
        ax.tick_params(axis='y', labelsize=FONTSIZE-2)
    plt.tight_layout()
    plt.savefig(f'Figures\mean_waiting_times_n{n}.pdf', bbox_inches='tight', format='pdf')
    plt.show()

# Set parameters lambda, mu, rho, number of simulations, max customers and starting value.
LAMBDA = 1
MU = 1.2
rho_list = [0.8, 0.9, 0.99]
NUM_SIM = 250
MAX_CUST = 1000
START = 11
n_list = [1, 2, 4]
for N in n_list:
    plot_mean_waiting_times(LAMBDA, MU, N, rho_list, NUM_SIM, MAX_CUST, START)
