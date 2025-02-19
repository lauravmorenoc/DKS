import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import correlate


def add_noise(signal, snr_db):
    """
    Add Gaussian noise to a signal based on a given SNR (dB).
    
    Parameters:
    - signal (numpy array): The original signal.
    - snr_db (float): Desired signal-to-noise ratio in dB.

    Returns:
    - noisy_signal (numpy array): Signal with added noise.
    """
    # Calculate signal power
    signal_power = np.mean(signal**2)

    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10)

    # Compute noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise
    
    return noisy_signal


# Simulate an unknown sinusoidal signal
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector for 1 second
f_real = 5  # Actual frequency of the signal (Hz) -> UNKNOWN in this case
I = 2  # Amplitude
Q = 2

# Generate the sinusoidal signal
raw_signal = I * np.cos(2 * np.pi * f_real * t) + Q * np.sin(2 * np.pi * f_real * t)
raw_signal=np.real(raw_signal)
sps=20
zeros=[0] * math.ceil(len(raw_signal)/2)
signal=np.append(zeros, raw_signal)
samples=np.append(signal,zeros)
t=np.arange(0,len(samples)/fs,1/fs)

# Add noise with an SNR of 10 dB
snr_db = 10
samples_noise = add_noise(samples, snr_db)

correlation = correlate(raw_signal, samples_noise, mode='full')

# Find the peak of the correlation (frame start)
peak_index = np.argmax(np.abs(correlation))

t=np.arange(0, len(samples_noise), 1)
zeros=[0]*len(samples_noise)
c=peak_index-len(raw_signal)+1
print(c)
zeros[c]=1

# Plot signals
plt.figure(figsize=(8, 4))
plt.plot(t, samples_noise, label="Signal")  # Plot the sinusoidal signal
#plt.scatter(t, samples_noise[peak_index], label="Signal")  # Plot the sinusoidal signal
plt.plot(t,zeros)
plt.show()
