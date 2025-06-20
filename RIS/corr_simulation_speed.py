import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import time

# Fixed parameters
NumSamples = 300000
window_size = 20
averaging_factor = 5
mult_factor=2**12

# === m-sequence ===
mseq = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0])
amp = 1
mseq = np.where(mseq == 0, amp, -amp)

# === Data simulation ===
sps=16
Rx = np.repeat(mseq, NumSamples/sps)
Rx=Rx*mult_factor
Rx=Rx+ 10 + np.abs(Rx)
Rx_data=Rx

'''Rx = Rx[::10]  # Downsample
envelope = np.abs(Rx)
envelope=envelope / mult_factor
envelope -= np.mean(envelope)
envelope /= np.max(envelope)

# Plotting sim data
plt.figure()
plt.plot(envelope)
plt.ylabel('Sim Data')
plt.grid(True)
plt.show()'''


# Values of downsample_factor to test
downsample_values = np.linspace(1, 180, 20, dtype=int)
execution_times = []

for downsample_factor in downsample_values:
    start_time = time.time()

    sps = 18750 // downsample_factor
    mseq_upsampled = np.repeat(mseq, sps)
    M_up = len(mseq_upsampled)

    corr_peaks = []
    corr_array = []

    for _ in range(window_size):
        Rx = Rx_data[::downsample_factor]  # Downsample
        envelope = np.abs(Rx) / mult_factor
        envelope -= np.mean(envelope)
        envelope /= np.max(envelope)
        corr = np.max(np.abs(correlate(mseq_upsampled, envelope, mode='full')) / M_up)

        corr_array.append(corr)
        if len(corr_array) > averaging_factor:
            corr_array = corr_array[-averaging_factor:]

        avg_corr = np.mean(corr_array)
        corr_peaks.append(avg_corr)

    end_time = time.time()
    execution_times.append(end_time - start_time)

# Plotting the results
plt.figure()
plt.plot(downsample_values, execution_times, marker='o')
plt.xlabel('Downsample Factor')
plt.ylabel('Execution Time (s)')
plt.title('Algorithm Speed vs. Downsample Factor')
plt.grid(True)
plt.show()
