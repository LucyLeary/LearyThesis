import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes.noise import (ColoredNoise)


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


def create_colored_noise_realization(beta, list, noisy_variable):
    cn = ColoredNoise(beta, T)
    list[0] = 0.5
    for i in range(n - 1):
        s = cn.sample(n)
        if noisy_variable == 'x':
            list[i + 1] = list[i] + dt * (r*list[i]*(1 - list[i]/K) - (a*list[i]**q)/(list[i]**q + h**q) + s[i])
        if noisy_variable == 'r':
            list[i + 1] = list[i] + dt * ((r + s[i]) * list[i] * (1 - list[i] / K) - (a * list[i] ** q) / (list[i] ** q + h ** q))
        if noisy_variable == 'K':
            list[i + 1] = list[i] + dt * (r * list[i] * (1 - list[i] / (K + 10 * s[i])) - (a * list[i] ** q) / (list[i] ** q + h ** q))
        if noisy_variable == 'a':
            list[i + 1] = list[i] + dt * (r * list[i] * (1 - list[i] / K) - ((a + s[i]) * list[i] ** q) / (list[i] ** q + h ** q))
        if noisy_variable == 'h':
            list[i + 1] = list[i] + dt * (r * list[i] * (1 - list[i] / K) - (a * list[i] ** q) / (list[i] ** q + (h + s[i]) ** q))
        if noisy_variable == 'q':
            list[i + 1] = list[i] + dt * (r * list[i] * (1 - list[i] / K) - (a * list[i] ** (q + 100 * s[i])) / (list[i] ** (q + 100 * s[i]) + h ** (q + 100 * s[i])))

def create_colored_noise_graph(beta, repetitions, noisy_variable):
    # Simulation of the process for 10 realizations with Blue Noise
    averages_list = np.zeros(n)
    for i in range(repetitions):
        single_list = np.zeros(n)
        create_colored_noise_realization(beta, single_list, noisy_variable)
        for j in range(n):
            averages_list[j] += single_list[j]
    for j in range(n):
        averages_list[j] = averages_list[j]/repetitions

    # Plot the single realization
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, averages_list, lw=2)
    ax.set_title('Beta = ' + str(beta) + " with " + str(repetitions) + ' realizations')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()


for number in range(-2, 3):
    for item in ['x', 'r', 'K', 'a', 'h', 'q']:
        create_colored_noise_graph(number, 10, item)

# For beta, 0 is white noise, 1 is pink noise, 2 is red noise (Brownian noise),
# -1 is blue noise, -2 is violet noise.

# Only seen it jump late once with variable K and beta = 2
