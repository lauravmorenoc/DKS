import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

def detect_frame_edges(signal, preamble):
    """
    Detects frame edges using cross-correlation with a known preamble.

    Parameters:
    - signal (numpy array): The received signal.
    - preamble (numpy array): The known preamble sequence.

    Returns:
    - peak_index (int): The estimated start of the frame.
    - correlation (numpy array): The cross-correlation result.
    """
    # Compute cross-correlation
    correlation = correlate(signal, preamble, mode='full')

    # Find the peak of the correlation (frame start)
    peak_index = np.argmax(np.abs(correlation))

    return peak_index, correlation

# Simulation Parameters
fs = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector

# Generate a known preamble (e.g., a short BPSK signal or PN sequence)
preamble_length = 50
preamble = np.sign(np.random.randn(preamble_length))  # Random BPSK preamble

# Generate a wireless signal containing the preamble
noise = np.random.randn(len(t)) * 0.1  # Additive White Gaussian Noise (AWGN)
frame_start = 300  # Define the frame start index
signal = noise.copy()
signal[frame_start:frame_start + preamble_length] += preamble  # Insert preamble

# Detect frame edges using cross-correlation
detected_index, correlation = detect_frame_edges(signal, preamble)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(signal, label="Received Signal")
plt.axvline(frame_start, color='g', linestyle="--", label="True Frame Start")
plt.axvline(detected_index - len(preamble) + 1, color='r', linestyle="--", label="Detected Frame Start")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Received Signal and Detected Frame Edge")

plt.subplot(2, 1, 2)
plt.plot(correlation, label="Cross-Correlation")
plt.axvline(detected_index, color='r', linestyle="--", label="Max Correlation (Frame Edge)")
plt.xlabel("Sample Index")
plt.ylabel("Correlation Value")
plt.legend()
plt.title("Cross-Correlation for Frame Synchronization")

plt.tight_layout()
plt.show()

# Print detected frame start
print(f"True Frame Start: {frame_start}")
print(f"Detected Frame Start: {detected_index - len(preamble) + 1}")
