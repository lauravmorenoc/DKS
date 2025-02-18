import numpy as np
import matplotlib.pyplot as plt
import math

'''Functions'''

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

def  Mueller_Muller_clock_recovery(samples):
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples[i_in] # grab what we think is the "best" sample
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    return out # only include this line if you want to connect this code snippet with the Costas Loop later on


# Simulate an unknown sinusoidal signal
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector for 1 second
f_real = 5  # Actual frequency of the signal (Hz) -> UNKNOWN in this case
I = 2  # Amplitude
Q = 2

# Generate the sinusoidal signal
raw_signal = I * np.cos(2 * np.pi * f_real * t) + Q * np.sin(2 * np.pi * f_real * t)
zeros=[0] * math.ceil(len(raw_signal)/2)

signal=np.append(zeros, raw_signal)
signal=np.append(signal,zeros)
t=np.arange(0,len(signal)/fs,1/fs)

# Add noise with an SNR of 10 dB
snr_db = 10
signal = add_noise(signal, snr_db)

# Use Müller and Muller alg.
syncronized_signal=Mueller_Muller_clock_recovery(signal)

# Plot signals
plt.figure(figsize=(8, 4))
plt.plot(t, signal, label="Signal")  # Plot the sinusoidal signal

plt.figure(figsize=(8, 4))
plt.plot(syncronized_signal, label="Signal")  # Plot the sinusoidal signal