import numpy as np
import matplotlib.pyplot as plt
import math

# Simulate an unknown sinusoidal signal
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector for 1 second
f_real = 5  # Actual frequency of the signal (Hz) -> UNKNOWN in this case
I = 2  # Amplitude
Q = 2

# Generate the sinusoidal signal
signal = I * np.cos(2 * np.pi * f_real * t) + Q * np.sin(2 * np.pi * f_real * t)

# Detect zero crossings with positive slope (start of a new cycle)
indices_zero = np.where((signal[:-1] < 0) & (signal[1:] > 0))[0]  # Condition for ascending zero crossing

# Estimate period and get peaks
periods = np.diff(indices_zero)  # Compute differences between successive indices
T_estimated = np.mean(periods) / fs  # Convert to time in seconds
indices_peaks=indices_zero[:-1]+math.ceil(np.mean(periods)/4)
print(indices_peaks)

# Plot the signal and detected points
plt.figure(figsize=(8, 4))
plt.plot(t, signal, label="Signal")  # Plot the sinusoidal signal
plt.scatter(t[indices_zero], signal[indices_zero], color='red', label="Zero crossings", zorder=3)  # Mark detected points
plt.scatter(t[indices_peaks], signal[indices_peaks], color='green', label="Peaks", zorder=3)  # Mark detected points
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")  # Reference zero line
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Cycle Restart Detection in the Signal")
plt.grid(True)
plt.show()

# Print the indices where the signal restarts its cycle
print("Indices where the signal restarts its cycle:", indices_zero)

# Estimate the period and frequency of the signal
periods = np.diff(indices_zero)  # Compute differences between successive indices
T_estimated = np.mean(periods) / fs  # Convert to time in seconds

print(f"Estimated period: {T_estimated:.4f} s")
print(f"Estimated frequency: {1/T_estimated:.2f} Hz")


''' This is additional and added by Laura '''
# Finding signal start and its real and imaginary part
Tx=signal
Tx_phased=Tx[indices_zero[0]:] # set the 0
t_phased=t[:len(Tx_phased)]
indices_peaks_phased=indices_peaks-indices_zero[0]

# Plot phased signal
plt.figure(figsize=(8, 4))
plt.plot(t, signal, label="Signal")  # Plot the sinusoidal signal
plt.plot(t_phased, Tx_phased, label="Phased signal")  # Plot the phased signal
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")  # Reference zero line
plt.xlabel("I")
plt.ylabel("Q")
plt.legend()
plt.title("Phased signal")
plt.grid(True)
plt.show()


# Constelation
#N=int(T_estimated*fs) # Number of samples in one period
N=len(Tx[indices_zero[0]:indices_zero[1]])
I_estimated = (2 / N) * np.sum(Tx_phased[:N] * np.cos(2 * np.pi * t_phased[:N] / T_estimated))  # Projection onto cos(x)
Q_estimated = (2 / N) * np.sum(Tx_phased[:N] * np.sin(2 * np.pi * t_phased[:N] / T_estimated))  # Projection onto sin(x)

# Print estimated vs real values
print(f"Real I: {I}, Estimated I: {I_estimated}")
print(f"Real Q: {Q}, Estimated Q: {Q_estimated}")

plt.figure(figsize=(8, 4))
plt.scatter(I_estimated, Q_estimated, color='blue', zorder=3)  # Mark detected points
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")  # Reference zero line
plt.xlabel("I")
plt.ylabel("Q")
plt.legend()
plt.title("Estimated symbol")
plt.grid(True)
plt.show()

