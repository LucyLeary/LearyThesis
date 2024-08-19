import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 1.  # Standard deviation.
mu = 10.  # Mean.
tau = .05  # Time constant.

# Time parameters
dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

# Adjusted parameters for the process
sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)

# Simulation of the process for a single realization
x = np.zeros(n)
for i in range(n - 1):
    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

# Plot the single realization
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('Single realization of the process')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
plt.show()

# Simulation of the process for multiple trials
ntrials = 10000
X = np.zeros(ntrials)

# Histogram bins
bins = np.linspace(-2., 14., 100)

# Create a new figure for plotting histograms
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# Loop through time steps
for i in range(n):
    # Update the process independently for all trials
    X += dt * (-(X - mu) / tau) + sigma_bis * sqrtdt * np.random.randn(ntrials)

    # Display the histogram for specific points in time
    if i in (5, 50, 900):
        hist, _ = np.histogram(X, bins=bins)
        ax.plot((bins[1:] + bins[:-1]) / 2, hist, {5: '-', 50: '.', 900: '-.', }[i], label=f"t={i * dt:.2f}")

# Add legend and labels to the plot
ax.legend()
ax.set_title('Histograms at different time points')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
plt.show()
