'''
This module contains code to plot simulated queues from the mmn_queue.py module.
'''
import numpy as np
import matplotlib.pyplot as plt
from mmn_queue import sims

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_mean_waiting_times(lambd, mu, n_list, rho_values, num_simulations, max_customers, starting_amount, method = 'fifo'):
    '''
    Plot the mean of the mean waiting times for a certain number of simulations. 
    '''
    FONTSIZE = 25
    COLORS = ['tab:blue', 'tab:orange', 'tab:green']  

    fig, axes = plt.subplots(1, len(rho_values), figsize=(len(rho_values)*5, 5), sharey=True)

    for i, n in enumerate(n_list):
        ax = axes[i]
        for j, rho in enumerate(rho_values):
            if method == 'tail':
                sim = sims(lambd, mu, n, rho=rho, tail=True)
            elif method == 'sjf':
                sim = sims(lambd, mu, n, rho=rho, sjf=True)
            elif method == 'deterministic':
                sim = sims(lambd, mu, n, rho=rho, deterministic=True)
            else:
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

            ax.fill_between(x_values, mean_values - std_values, mean_values + std_values, color=COLORS[j], alpha=0.2, label=rf'$\rho={rho}$')
            ax.plot(x_values, mean_values, color=COLORS[j])

            ax.axhline(np.mean(mean_values), color=COLORS[j], linestyle='--')
            
            if ax == axes[0]:
                ax.set_ylabel('Mean Waiting Time', fontsize=FONTSIZE)
                ax.legend(loc = 'upper left', fontsize=FONTSIZE-4)
            if ax == axes[1]:
                ax.set_xlabel('Number of Simulations', fontsize=FONTSIZE)
        
            ax.text(0.95, 0.95, f'n ={n}', transform=ax.transAxes, fontsize=FONTSIZE-3,
            verticalalignment='top', horizontalalignment='right')
            ax.tick_params(axis='x', labelsize=FONTSIZE-4)
            ax.tick_params(axis='y', labelsize=FONTSIZE-4, )
            if method == 'fifo':
                ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40])

    plt.tight_layout()
    plt.savefig(f'Figures\mean_waiting_times_{method}.pdf', bbox_inches='tight', format='pdf')
    plt.show()


# Set parameters lambda, mu, rho, number of simulations, max customers and starting value.
LAMBDA = 1
MU = 1.2
rho_list = [0.8, 0.9, 0.99]
NUM_SIM = 250
MAX_CUST = 1000
START = 10
n_list = [1, 2, 4]
plot_mean_waiting_times(LAMBDA, MU, n_list, rho_list, NUM_SIM, MAX_CUST, START) # Plot for fifo
plot_mean_waiting_times(LAMBDA, MU, n_list, rho_list, NUM_SIM, MAX_CUST, START, method='sjf') # Plot for sjf
