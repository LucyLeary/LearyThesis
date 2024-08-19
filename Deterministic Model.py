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

# Simulation of the process for a single realization with no noise
x = np.zeros(n)
x[0] = 0.5
for i in range(n - 1):
    x[i + 1] = x[i] + dt * (r*x[i]*(1 - x[i]/K) - (a*x[i]**q)/(x[i]**q + h**q))

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Deterministic')
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()
