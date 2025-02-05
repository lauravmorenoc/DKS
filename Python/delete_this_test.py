import numpy as np

# Sampling parameters
fs = 1000  # Sampling frequency (Hz)
T = 1 / 5  # Period of the signal (assuming frequency is unknown)
N = int(T * fs)  # Number of samples in one period

# Time vector
t = np.linspace(0, T, N, endpoint=False)

# Signal parameters (real values of I and Q)
I_real, Q_real = 3, 2  # In-phase and Quadrature components

# Generate the received signal: signal = I*cos(x) + Q*sin(x)
signal = I_real * np.cos(2 * np.pi * t / T) + Q_real * np.sin(2 * np.pi * t / T)

# Estimate I and Q using discrete integration (sum approximation)
I_estimated = (2 / N) * np.sum(signal * np.cos(2 * np.pi * t / T))  # Projection onto cos(x)
Q_estimated = (2 / N) * np.sum(signal * np.sin(2 * np.pi * t / T))  # Projection onto sin(x)

# Print estimated vs real values
print(f"Real I: {I_real}, Estimated I: {I_estimated}")
print(f"Real Q: {Q_real}, Estimated Q: {Q_estimated}")