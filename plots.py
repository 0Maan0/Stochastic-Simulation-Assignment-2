"""
This module visualizes the performance of simulated queuing systems 
based on the 'mmn_queue.py' module. Including code to plot mean waiting 
times across multiple simulations for different configurations of 
queueing systems and a plot of SFJ vs FIFO scheduling systems for the 
same parameters.

Key Features:
    - Generates plots to compare the effect of different server counts 
      (n) and system load factors (ρ) on mean waiting times.
    - Supports various queuing methods, including FIFO 
      (First-In-First-Out), SJF (Shortest Job First), deterministic 
      service times (M/D/1), and long-tail service time distributions.

Usage Example:
    - Run the module to generate plots for FIFO and SJF queueing methods.
    - Plots are saved to the `Figures/` directory in PDF format.

Dependencies:
    - numpy: Numerical computations.
    - matplotlib: Plotting.
    - mmn_queue: Sims class that handles queue simulations.

The generated plots display:
    - Mean waiting times with standard deviation bands.
    - Subplots for different server configurations (n), with legends for 
      various 'ρ' values.
"""


import numpy as np
import matplotlib.pyplot as plt
from mmn_queue import sims

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


def plot_mean_waiting_times(lam, mu, n_list, rho_values, num_simulations, 
                            max_customers, starting_amount, method = 'fifo'):
    """
    Plot the mean waiting times across multiple simulations for 
    different system configurations.

    Parameters:
        - lam (float): Arrival rate (λ) of customers into the system.
        - mu (float): Service rate (μ) of servers.
        - n_list (list[int]): List of server counts (`n`) to simulate.
        - rho_values (list[float]): List of load factors (`ρ`) to 
          simulate.
        - num_simulations (int): Total number of simulations to run for 
          each configuration.
        - max_customers (int): Maximum number of customers per 
          simulation.
        - starting_amount (int): Number of initial simulations to 
          exclude from statistical analysis.
        - method (str, optional): Queueing method. Options are:
            - 'fifo': First-In-First-Out (default).
            - 'sjf': Shortest Job First.
            - 'deterministic': Deterministic service times (M/D/1).
            - 'tail': Long-tail (hyperexponential) service time 
              distribution.

    Outputs:
        A PDF plot, saved to `Figures/` directory, showing mean waiting 
        times and their standard deviations as shaded regions for 
        different values of 'ρ' and 'n'.

    Raises:
        ValueError: If an invalid `method` is provided.
    """

    FONTSIZE = 25
    COLORS = ['tab:blue', 'tab:orange', 'tab:green']  

    fig, axes = plt.subplots(1, len(rho_values), 
                             figsize=(len(rho_values)*5, 5), sharey=True)

    for i, n in enumerate(n_list):
        ax = axes[i]

        #Perform simulations according to called method.
        for j, rho in enumerate(rho_values):
            if method == 'tail':
                sim = sims(lam, mu, n, rho=rho, tail=True)

            elif method == 'sjf':
                sim = sims(lam, mu, n, rho=rho, sjf=True)

            elif method == 'deterministic':
                sim = sims(lam, mu, n, rho=rho, deterministic=True)

            else:
                sim = sims(lam, mu, n, rho=rho)

            #Generate lists for plotting.
            mean_waiting_times = []
            mean_values = []
            std_values = []

            for _ in range(num_simulations):
                simulation = sim.simulate_queue(max_customers=max_customers)
                mean_waiting_times.append(np.mean(simulation))

                #Calculate mean and std after starting_amount simulations.
                if len(mean_waiting_times) >= starting_amount:
                    current_mean = np.mean(mean_waiting_times)
                    current_std = np.std(mean_waiting_times, ddof=1)
                    mean_values.append(current_mean)
                    std_values.append(current_std)

            #X-array for number of simulations.
            x_values = np.linspace(starting_amount, num_simulations, 
                                   num=len(mean_values))
            mean_values = np.array(mean_values)
            std_values = np.array(std_values)

            ax.fill_between(x_values, mean_values - std_values, mean_values 
                            + std_values, color=COLORS[j], alpha=0.2, 
                            label=rf'$\rho={rho}$')
            ax.plot(x_values, mean_values, color=COLORS[j])

            ax.axhline(np.mean(mean_values), color=COLORS[j], linestyle='--')
            
            if ax == axes[0]:
                ax.set_ylabel('Mean Waiting Time', fontsize=FONTSIZE)
                ax.legend(loc = 'upper left', fontsize=FONTSIZE-4)

            if ax == axes[1]:
                ax.set_xlabel('Number of Simulations', fontsize=FONTSIZE)
        
            ax.text(0.95, 0.95, f'n ={n}', transform=ax.transAxes, 
            fontsize=FONTSIZE-3, verticalalignment='top', 
            horizontalalignment='right')
            ax.tick_params(axis='x', labelsize=FONTSIZE-4)
            ax.tick_params(axis='y', labelsize=FONTSIZE-4)
            if method == 'fifo':
                ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40])

    plt.tight_layout()
    plt.savefig(f'Figures\mean_waiting_times_{method}.pdf', 
                bbox_inches='tight', format='pdf')
    plt.show()


def plot_sfj_vs_fifo(lam, mu, n, rho=0.9):
    """
    Compare waiting times in SJF and FIFO scheduling systems.

    Parameters:
        - lam (float): Arrival rate (λ) of customers into the system.
        - mu (float): Service rate (μ) of servers.
        - n (int): Number of servers in the system.
        - rho (float, optional): System load factor (ρ). Default is 0.9.

    Outputs:
        - A PDF plot saved to the 'Figures/' directory with the filename 
          'sjf_vs_fifo_n{n}_rho{rho}.pdf'.
        The plot contains two subplots:
            - Left: SJF waiting times vs. service times.
            - Right: FIFO waiting times vs. service times.
    """

    #Perform SJF simulation.
    sim_sjf = sims(lam, mu, n, rho, sjf=True)
    sim_sjf.simulate_queue()

    #Perform FIFO simulation.
    sim_fifo = sims(lam, mu, n, rho, sjf=False)
    sim_fifo.simulate_queue()

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True,
                           gridspec_kw={'wspace': 0.05})

    ax[0].scatter(sim_sjf.service_time_list, sim_sjf.waiting_times, marker='.',
                  label='SJF', s=50, lw=0)
    ax[1].scatter(sim_fifo.service_time_list, sim_fifo.waiting_times, 
                  marker='.', label='FIFO', s=50, lw=0)
    ax[0].set_ylabel(r'Waiting Time ($t_w$)')
    ax[0].set_xlabel(r'Service Time ($t_s$)')
    ax[1].set_xlabel(r'Service Time ($t_s$)')
    ax[0].legend()
    ax[1].legend()

    plt.savefig(
        f'Figures/sjf_vs_fifo_n{n}_rho{rho}.pdf',
        bbox_inches='tight', format='pdf'
    )
    plt.show()

    return


#Set appropriate parameters.
LAMBDA = 1
MU = 1.2
rho_list = [0.8, 0.9, 0.99]
NUM_SIM = 250
MAX_CUST = 1000
START = 10
n_list = [1, 2, 4]

lam = 1
mu = 1.2
n = 1

#Plot for FIFO.
plot_mean_waiting_times(LAMBDA, MU, n_list, rho_list, NUM_SIM, MAX_CUST, START)

#Plot for SJF.
plot_mean_waiting_times(LAMBDA, MU, n_list, rho_list, NUM_SIM, MAX_CUST, 
                        START, method='sjf')

#Plot SFJ vs FIFO.
plot_sfj_vs_fifo(lam, mu, n, rho=0.9)
