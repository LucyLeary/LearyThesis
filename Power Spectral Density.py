import scipy
from stochastic.processes.noise import (ColoredNoise)
import matplotlib.pyplot as plt


betas = [-2, -1, 0, 1, 2]
for beta in betas:
    cn = ColoredNoise(beta, 1)
    s = cn.sample(100000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    data = scipy.signal.welch(s)
    x, y = data[0], data[1]
    # print(scipy.signal.welch(s))
    ax.plot(x,y)

    for i in range(99999):
        s[i] = s[i] * 2

    data = scipy.signal.welch(s)
    x, y = data[0], data[1]
    # print(scipy.signal.welch(s))
    ax.plot(x, y, label="scaled by 2")
    plt.legend()


    plt.show()