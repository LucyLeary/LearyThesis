import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes.noise import (FractionalGaussianNoise)


# Jody's Parameters
r = 0.05
K = 2
a = 0.023
h = 0.38
q = 5

# Time parameters
dt = 1  # Time step.
T = 2500.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.


def create_fractional_gaussian_realization(list):
    fgn = FractionalGaussianNoise(0.5, T)
    list[0] = 0.5
    for i in range(n - 1):
        s = fgn.sample(n)
        list[i + 1] = list[i] + dt * (r*list[i]*(1 - list[i]/K) - (a*list[i]**q)/(list[i]**q + h**q)) + s[i]


def create_fractional_gaussian_graph(repetitions):
    # Simulation of the process for 10 realizations with Blue Noise
    averages_list = np.zeros(n)
    for i in range(repetitions):
        single_list = np.zeros(n)
        create_fractional_gaussian_realization(single_list)
        for j in range(n):
            averages_list[j] += single_list[j]
    for j in range(n):
        averages_list[j] = averages_list[j]/repetitions

     # Plot the single realization
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, averages_list, lw=2)
    ax.set_title('Fractional Gaussian with ' + str(repetitions) + ' realizations')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

create_fractional_gaussian_graph(20)