from stochastic.processes.noise import (ColoredNoise)
import numpy as np
import matplotlib.pyplot as plt
import allantools

# Jody's Parameters
r = 0.05
K = 2
a = 0.023
h = 0.38
q = 5

# Time parameters
dt = 1  # Time step.
# T = 2500.  # Total time.
# n = int(T / dt)  # Number of time steps.
# t = np.linspace(0., T, n)  # Vector of times.


def plot_colored_noise(beta, T):
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    noise = ColoredNoise(beta, T)
    s = noise.sample(n)

    noise_list = np.zeros(n)
    for i in range(n - 1):
        noise_list[i] = s[i]

    ax.plot(t, noise_list, color="blue", lw=.5)
    ax.set_title('Graph of noise with beta = ' + str(beta))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig('colored_noise_beta_' + str(beta) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    betas = [-2, -1, 0, 1, 2]
    for beta in betas:
        cn = ColoredNoise(beta, 1)
        s = cn.sample(100)
        print("Allan variance with beta = " + str(beta))
        print(allantools.oadev(s, taus=[2]))  # calculating variance

        for i in range(99):
            s[i] = s[i]*2
        print("SCALED BY 2:")
        print(allantools.oadev(s, taus=[2]))


    # for beta in betas:
    #     cn = ColoredNoise(beta, 1)
    #     s = cn.sample(1000)
    #     print("Variance with beta = " + str(beta))
    #     print(np.var(s))

    # plot_colored_noise(0, 4000)
    # plot_colored_noise(0, 3000)
    # plot_colored_noise(0, 2000)
    # plot_colored_noise(0, 1000)
    # plot_colored_noise(0, 500)
    # plot_colored_noise(0, 50)
    #
    # plot_colored_noise(0, 4000)
    # plot_colored_noise(0, 3000)
    # plot_colored_noise(0, 2000)
    # plot_colored_noise(0, 1000)
    # plot_colored_noise(0, 500)
    # plot_colored_noise(0, 50)

if __name__ == "__main__":
    main()