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
            realization[i + 1] = realization[i] + dt * (r * realization[i] * (1 - realization[i] / K) - (a * realization[i] ** q) / (realization[i] ** q + h ** q) + scaling_factor * s[i])
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


def create_mfpt_plot(beta, noisy_variable, scaling_factor, repetitions, average_passages, errors, fig, position):

    axis1 = fig.add_subplot(position)

    realizations = [create_realization(beta, noisy_variable, scaling_factor) for _ in range(repetitions)]
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    passages = [get_passage(realization) for realization in realizations]

    error_count = sum([1 for p in passages if p is None])
    print(f"error count: {error_count}")
    average_passage = sum([p for p in passages if p is not None]) / (repetitions - error_count)

    average_passages.append(average_passage)

    averages_list = [
        sum([realization[i] for realization in realizations]) / repetitions
        for i in range(n)
    ]

    for realization in realizations:
        axis1.plot(t, realization, c="gray", lw=0.5)

    axis1.plot(t, averages_list, color="black", lw=1)
    det = plt.axvline(x=1585, color='red', label="deterministic MFPT")
    stoch = plt.axvline(x=average_passage, color='blue', label="stochastic MFPT")
    axis1.set_title('Beta = ' + str(beta))
    axis1.set_xlabel('t')
    axis1.set_ylabel('x')
    print(average_passage)
    errors.append(error_count)
    return [det, stoch]


def print_variance(beta):
    cn = ColoredNoise(beta, T)
    s = cn.sample(n)
    print("Variance with beta = " + str(beta))
    print(np.var(s))  # calculating variance

def save_plots_to_one_file(betas, noisy_variable, scaling_factor, repetitions, average_passages, errors):
    fig = plt.figure(figsize=(10.0, 5.0))
    position = 230
    for beta in betas:
        position += 1
        legend_position = create_mfpt_plot(beta, noisy_variable, scaling_factor, repetitions, average_passages, errors, fig, position)
    fig.legend(legend_position, ["deterministic MFPT", "stochastic MFPT"], bbox_to_anchor=(.92, .45))
    fig.tight_layout()
    fig.savefig('repetitions_' + str(repetitions) + '_parameter_' + noisy_variable + '_scaled_by_' + str(
            scaling_factor) + '.png', bbox_inches='tight')

def main():
    # Customizations
    repetitions = 10000
    noisy_variable = 'K'
    scaling_factor = 5.4
    betas = [-2, -1, 0, 1, 2]

    # Create plots and a table comparing MFPT and beta
    average_passages = []
    errors = []

    save_plots_to_one_file(betas, noisy_variable, scaling_factor, repetitions, average_passages, errors)

    table = pd.DataFrame({'mean first passage': average_passages,
                         'beta': betas, 'error count': errors})
    print(table)
    table.to_csv(noisy_variable + '_' + str(scaling_factor) + '.csv', index=False)

if __name__ == "__main__":
    main()
