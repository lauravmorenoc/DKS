import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def zero_crossings(signal):
    """
    Detect zero-crossings in the derivative of a signal.
    
    Parameters:
    - signal (numpy array): Input signal.

    Returns:
    - indices (numpy array): Indices where zero-crossings occur.
    """
    # Compute the first derivative (rate of change)
    derivative = np.diff(signal)

    # Find zero-crossings: where derivative changes sign
    zero_crossing_indices = np.where(np.diff(np.sign(derivative)))[0]

    return zero_crossing_indices

# Simulated wireless signal (modulated burst with noise)
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector

# Create a modulated signal with a frame transition
carrier = np.cos(2 * np.pi * 50 * t)  # 50 Hz carrier
modulation = np.hstack([np.ones(400), np.zeros(200), np.ones(400)])  # Frame structure
signal = modulation * carrier  # Modulated signal

# Extract the envelope using the Hilbert Transform
envelope = np.abs(hilbert(signal))

# Detect frame edges using zero-crossings in the envelope's derivative
frame_edges = zero_crossings(envelope)

v=envelope[frame_edges]
v_index=np.where(v<1)[0]  # Condition for ascending zero crossing


# Plot the signal and detected frame edges
plt.figure(figsize=(10, 5))
plt.plot(t, envelope, label="Signal Envelope", linewidth=2)
plt.scatter(t[v_index], envelope[v_index], color='red', label="Detected Frame Edges", zorder=3)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Frame Edge Detection Using Zero-Crossing Method")
plt.grid(True)
plt.show()
