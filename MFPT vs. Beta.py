import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stochastic.processes.noise import (ColoredNoise)

# Jody's Parameters
r = 0.05
K = 2
a = 0.023
h = 0.38
q = 5

# Time parameters
dt = 1  # Time step.
T = 4000.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.


# Creates single realization of stochastic model
def create_realization(beta, noisy_variable, scaling_factor):
    realization = np.zeros(n)
    realization[0] = 0.5
    cn = ColoredNoise(beta, T)
    s = cn.sample(n)
    if noisy_variable == 'x':
        for i in range(n - 1):
            realization[i + 1] = realization[i] + dt * (r * realization[i] * (1 - realization[i] / K) - (a * realization[i] ** q) / (realization[i] ** q + h ** q)) + scaling_factor * s[i]
    if noisy_variable == 'r':
        for i in range(n - 1):
            realization[i + 1] = realization[i] + dt * ((r + scaling_factor * s[i]) * realization[i] * (1 - realization[i] / K) - (a * realization[i] ** q) / (realization[i] ** q + h ** q))
    if noisy_variable == 'K':
        for i in range(n - 1):
            realization[i + 1] = realization[i] + dt * (r * realization[i] * (1 - realization[i] / (K + scaling_factor * s[i])) - (a * realization[i] ** q) / (realization[i] ** q + h ** q))
    if noisy_variable == 'a':
        for i in range(n - 1):
            realization[i + 1] = realization[i] + dt * (r * realization[i] * (1 - realization[i] / K) - ((a + scaling_factor * s[i]) * realization[i] ** q) / (realization[i] ** q + h ** q))
    if noisy_variable == 'h':
        for i in range(n - 1):
            realization[i + 1] = realization[i] + dt * (r * realization[i] * (1 - realization[i] / K) - (a * realization[i] ** q) / (realization[i] ** q + (h + scaling_factor * s[i]) ** q))
    if noisy_variable == 'q':
        for i in range(n - 1):
            realization[i + 1] = realization[i] + dt * (r * realization[i] * (1 - realization[i] / K) - (a * realization[i] ** (q + 100 * s[i])) / (realization[i] ** (q + scaling_factor * s[i]) + h ** (q + 100 * s[i])))
    return realization


def get_passage(realization):
    try:
        return next(idx for idx, elem in enumerate(realization) if elem >= 0.9798)
    except StopIteration:
        return None


def create_mfpt_beta_plot(noisy_variable, scaling_factors, repetitions):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    average_passages = []
    betas = [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]

    for scaling_factor in scaling_factors:
        average_passages.clear()
        for beta in betas:
            realizations = [create_realization(beta, noisy_variable, scaling_factor) for _ in range(repetitions)]
            passages = [get_passage(realization) for realization in realizations]

            error_count = sum([1 for p in passages if p is None])
            print(f"error count: {error_count}")
            average_passage = sum([p for p in passages if p is not None]) / (repetitions - error_count)
            average_passages.append(average_passage)

        ax.plot(betas, average_passages, label=str(scaling_factor))

    ax.set_title('Mean first passage time for parameter ' + noisy_variable)
    ax.set_xlabel('Beta')
    ax.set_ylabel('MFPT')
    plt.legend()

    fig.savefig('02.10.2025_MFPT_vs_BETA_repetitions_' + str(repetitions) + '_parameter_' + noisy_variable + '.png')

    plt.show()


def main():
    # Customizations
    repetitions = 10000
    noisy_variable = 'x'
    scaling_factors = [.01, .05, .1, .2, .3, .5]

    create_mfpt_beta_plot(noisy_variable, scaling_factors, repetitions)


if __name__ == "__main__":
    main()
