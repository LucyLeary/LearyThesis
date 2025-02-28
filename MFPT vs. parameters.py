import numpy as np
import matplotlib.pyplot as plt


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

sample_size = 50000

# Linspaces
r_array = np.linspace(0.04999, 0.05001, sample_size)
K_array = np.linspace(1.999, 2.001, sample_size)
a_array = np.linspace(0.02299, 0.02301, sample_size)
h_array = np.linspace(0.3799, 0.3801, sample_size)

def create_mfpt_param_plot(parameter):
    first_passages = []
    parameter_array = []
    if parameter == 'a':
        parameter_array = a_array
        for j in range(sample_size):
            # Simulation of the process for a single realization with no noise
            x = np.zeros(n)
            first_passage = False
            passage_index = 0
            x[0] = 0.5
            for i in range(n - 1):
                x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - ((a_array[j])*x[i]**q)/(x[i]**q + h**q))
                if first_passage == False and x[i] >= 0.9798:
                    passage_index = i
                    first_passage = True
            first_passages.append(passage_index)

    if parameter == 'r':
        parameter_array = r_array
        for j in range(sample_size):
            # Simulation of the process for a single realization with no noise
            x = np.zeros(n)
            first_passage = False
            passage_index = 0
            x[0] = 0.5
            for i in range(n - 1):
                x[i + 1] = x[i] + dt * (
                            r_array[j] * x[i] * (1 - x[i] / K) - ((a) * x[i] ** q) / (x[i] ** q + h ** q))
                if first_passage == False and x[i] >= 0.9798:
                    passage_index = i
                    first_passage = True
            first_passages.append(passage_index)

    if parameter == 'K':
        parameter_array = K_array
        for j in range(sample_size):
            # Simulation of the process for a single realization with no noise
            x = np.zeros(n)
            first_passage = False
            passage_index = 0
            x[0] = 0.5
            for i in range(n - 1):
                x[i + 1] = x[i] + dt * (
                            r * x[i] * (1 - x[i] / K_array[j]) - ((a) * x[i] ** q) / (x[i] ** q + h ** q))
                if first_passage == False and x[i] >= 0.9798:
                    passage_index = i
                    first_passage = True
            first_passages.append(passage_index)

    if parameter == 'h':
        parameter_array = h_array
        for j in range(sample_size):
            # Simulation of the process for a single realization with no noise
            x = np.zeros(n)
            first_passage = False
            passage_index = 0
            x[0] = 0.5
            for i in range(n - 1):
                x[i + 1] = x[i] + dt * (
                            r * x[i] * (1 - x[i] / K) - ((a) * x[i] ** q) / (x[i] ** q + h_array[j] ** q))
                if first_passage == False and x[i] >= 0.9798:
                    passage_index = i
                    first_passage = True
            first_passages.append(passage_index)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(parameter_array, first_passages, c=(.1, .7, .3), lw=2)

    ax.set_title('First passage time for parameter ' + parameter)
    ax.set_xlabel(parameter)
    ax.set_ylabel('First Passage')
    fig.tight_layout()
    # fig.savefig('02.24.2025_MFPT_vs_BETA_repetitions_' + str(repetitions) + '_parameter_' + noisy_variable + '.png')

    plt.show()


def main():
    # Customizations
    # for param in ['a', 'r', 'h', 'K']:
    for param in ['r', 'h', 'K']:
        create_mfpt_param_plot(param)


if __name__ == "__main__":
    main()


