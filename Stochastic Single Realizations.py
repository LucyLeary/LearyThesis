import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes.noise import (PinkNoise, BlueNoise, RedNoise, BrownianNoise,
                                        ColoredNoise, VioletNoise, WhiteNoise)


# Jody's Parameters
r = 0.05
K = 2
a = 0.023
h = 0.38
q = 5
# sigma = 0.02

# Time parameters
dt = 1  # Time step.
T = 2500.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

# Simulation of the process for a single realization with Pink Noise
x = np.zeros(n)
pn = PinkNoise(T)
x[0] = 0.5
for i in range(n - 1):
    s = pn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Pink')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Graph with Blue Noise
# Simulation of the process for a single realization
x = np.zeros(n)
bn = BlueNoise(T)
x[0] = 0.5
for i in range(n - 1):
    s = bn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Blue')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Graph with Red Noise
# Simulation of the process for a single realization
x = np.zeros(n)
rn = RedNoise(T)
x[0] = 0.5
for i in range(n - 1):
    s = rn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Red')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Graph with Brownian Noise
# Simulation of the process for a single realization
x = np.zeros(n)
bn = BrownianNoise(T)
x[0] = 0.5
for i in range(n - 1):
    s = bn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Brownian (red?)')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Graph with Colored Noise
# Simulation of the process for a single realization
x = np.zeros(n)
bn = ColoredNoise(0,T)
x[0] = 0.5
for i in range(n - 1):
    s = bn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Colored')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Graph with Violet Noise
# Simulation of the process for a single realization
x = np.zeros(n)
bn = VioletNoise(T)
x[0] = 0.5
for i in range(n - 1):
    s = bn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Violet')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Graph with White Noise
# Simulation of the process for a single realization
x = np.zeros(n)
bn = WhiteNoise(T)
x[0] = 0.5
for i in range(n - 1):
    s = bn.sample(n)
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q)) + s[i]

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('White')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()
