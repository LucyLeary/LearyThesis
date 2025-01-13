import time

import asyncio
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


def timer(func):
    def _timer(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        print(func.__name__, end - start)
        return result

    return _timer

def create_colored_noise_sample(beta):
    cn = ColoredNoise(beta, T)
    return cn

def create_colored_noise_realization(beta, sample, noisy_variable, scaling_factor):
    cn = create_colored_noise_sample(beta)
    sample[0] = 0.5
    for i in range(n - 1):
        s = cn.sample(n)
        if noisy_variable == 'x':
            sample[i + 1] = sample[i] + dt * (r * sample[i] * (1 - sample[i] / K) - (a * sample[i] ** q) / (sample[i] ** q + h ** q) + scaling_factor * s[i])
        if noisy_variable == 'r':
            sample[i + 1] = sample[i] + dt * ((r + scaling_factor * s[i]) * sample[i] * (1 - sample[i] / K) - (a * sample[i] ** q) / (sample[i] ** q + h ** q))
        if noisy_variable == 'K':
            sample[i + 1] = sample[i] + dt * (r * sample[i] * (1 - sample[i] / (K + scaling_factor * s[i])) - (a * sample[i] ** q) / (sample[i] ** q + h ** q))
        if noisy_variable == 'a':
            sample[i + 1] = sample[i] + dt * (r * sample[i] * (1 - sample[i] / K) - ((a + scaling_factor * s[i]) * sample[i] ** q) / (sample[i] ** q + h ** q))
        if noisy_variable == 'h':
            sample[i + 1] = sample[i] + dt * (r * sample[i] * (1 - sample[i] / K) - (a * sample[i] ** q) / (sample[i] ** q + (h + scaling_factor * s[i]) ** q))
        if noisy_variable == 'q':
            sample[i + 1] = sample[i] + dt * (r * sample[i] * (1 - sample[i] / K) - (a * sample[i] ** (q + 100 * s[i])) / (sample[i] ** (q + scaling_factor * s[i]) + h ** (q + 100 * s[i])))

def plot_colored_noise(beta):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    noise = create_colored_noise_sample(beta)
    s = noise.sample(n)

    noise_list = np.zeros(n)
    for i in range(n - 1):
        noise_list[i] = s[i]

    ax.plot(t, noise_list, color="blue", lw=.5)
    ax.set_title('Graph of noise with beta = ' + str(beta))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig('colored_noise_beta_' + str(beta) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_noisy_parameter (beta, noisy_variable):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    noise = create_colored_noise_sample(beta)
    s = noise.sample(n)
    noise_list = np.zeros(n)
    for i in range(n):
        noise_list[i] = s[i]

        if noisy_variable == 'r':
            noise_list[i] = (r + .1*s[i])
        if noisy_variable == 'K':
            noise_list[i] = (K + s[i])
        if noisy_variable == 'a':
            noise_list[i] = (a + .1*s[i])

    ax.plot(t, noise_list, color="purple", lw=.5)
    ax.set_title('Graph of noisy parameter ' + noisy_variable + ' with beta = ' + str(beta))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig('noisy_parameter_' + noisy_variable + '_beta_' + str(beta) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

def single_realization(beta, noisy_variable, scaling_factor):
    single_list = np.zeros(n)
    create_colored_noise_realization(beta, single_list, noisy_variable, scaling_factor)
    return single_list

def realizations(beta, repetitions, noisy_variable, scaling_factor):
    # tasks = [
    #     single_realization(beta, noisy_variable, scaling_factor)
    #     for i in range(repetitions)
    # ]
    # return await asyncio.gather(*tasks)
    return [single_realization(beta, noisy_variable, scaling_factor) for _ in range(repetitions)]


async def create_colored_noise_graph(beta, repetitions, noisy_variable, scaling_factor):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # Simulation of the process for 10 realizations with Blue Noise
    # averages_list = np.zeros(n)
    # sum_passage = 0

    # lists = asyncio.run(realizations(beta, repetitions, noisy_variable, scaling_factor))
    lists = realizations(beta, repetitions, noisy_variable, scaling_factor)

    def _get_passage(lst):
        try:
            return next(idx for idx, elem in enumerate(lst) if elem >= 0.9798)
        except StopIteration:
            return None

    passages = [_get_passage(l) for l in lists]

    error_count = sum([1 for p in passages if p is None])
    print(f"error count: {error_count}")
    average_passage = sum([p for p in passages if p is not None]) / (repetitions - error_count)

    averages_list = [
        sum([l[i] for l in lists]) / repetitions
        for i in range(n)
    ]

    for l in lists:
        ax.plot(t, l, c = "gray", lw=0.5)


    # for i in range(repetitions):
    #     first_passage_found = False
    #     single_list = np.zeros(n)
    #     create_colored_noise_realization(beta, single_list, noisy_variable, scaling_factor)
    #     ax.plot(t, single_list, c = "gray", lw=0.5)
    #     for j in range(n):
    #         averages_list[j] += single_list[j]
    #         if first_passage_found == False and single_list[j] >= 0.9798:
    #             sum_passage += j
    #             first_passage_found = True
    # for j in range(n):
    #     averages_list[j] = averages_list[j]/repetitions
    # average_passage = sum_passage / repetitions
    # Plot the single realization

    ax.plot(t, averages_list, color = "black", lw=1)
    # ax.plot(average_passage, 0.9798, 'o', color="red", lw = 10)
    det =  plt.axvline(x = 1585, color = 'red', label = "deterministic MFPT")
    stoch = plt.axvline(x=average_passage, color='blue', label = "stochastic MFPT")
    plt.legend([det, stoch], ["deterministic MFPT", "stochastic MFPT"])
    ax.set_title('Beta = ' + str(beta) + " with " + str(repetitions) + ' realizations and noisy parameter ' + noisy_variable + '_scaled_by_' + str(scaling_factor))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig('beta_' + str(beta) + '_repetitions_' + str(repetitions) + '_parameter_' + noisy_variable + '_scaled_by_' + str(scaling_factor) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    print(average_passage)


async def main(betas, repetitions, scaling_factors):
    tasks = [
        create_colored_noise_graph(number, repetitions, 'K', scaling_factor)
        for number in range(betas)
        for scaling_factor in scaling_factors
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main([-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2], 10, [5.4]))
# for number in range(-2, 3):
#     # plot_colored_noise(number)
#     # plot_noisy_parameter (number, 'K')
#     # for scalar in 5.5:
#     create_colored_noise_graph(number, 100, 'K', 5.4)
#     # create_colored_noise_graph(number, 1000, 'K', 5.3)
#     # create_colored_noise_graph(number, 1000, 'K', 5.2)
#     # create_colored_noise_graph(number, 1000, 'K', 5.1)
#     # create_colored_noise_graph(number, 1000, 'K', 5.0)
# # For beta, 0 is white noise, 1 is pink noise, 2 is red noise (Brownian noise),
# # -1 is blue noise, -2 is violet noise.