import adi  # Interface for ADALM-Pluto SDR
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

'''======== GUI for Binary RIS State ========'''
class BinaryStateVisualizer:
    def __init__(self):
        self.states = [0, 0]  # Initial binary states of RIS 1 and RIS 2

        # Create 2 colored circles to represent RIS states
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        centers = [(-0.3, -0.5), (1.3, -0.5)]
        labels = ["RIS 1", "RIS 2"]
        self.circles = [plt.Circle(center, 0.3, fc='red', edgecolor='black') for center in centers]

        for circle in self.circles:
            self.ax.add_patch(circle)

        for (x, y), label in zip(centers, labels):
            self.ax.text(x, y + 0.6, label, ha='center', va='center', fontsize=12, fontweight='bold')

        self.ax.set_xlim(-1, 2)
        self.ax.set_ylim(-1.5, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        plt.ion()  # Enables dynamic update
        plt.show()

    def update_states(self, new_states):
        """Update the RIS circles based on binary state."""
        if len(new_states) == 2:
            self.states = new_states
            for i, circle in enumerate(self.circles):
                circle.set_facecolor('green' if self.states[i] else 'red')
            self.fig.canvas.draw()
            plt.pause(0.1)

'''======== Configure Pluto SDR ========='''
def conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain, buffer_size):
    """Configure Pluto SDR reception settings"""
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(fc0 * 3)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(buffer_size)
    sdr._rxadc.set_kernel_buffers_count(1)
    fs = int(sdr.sample_rate)
    ts = 1 / fs
    return [fs, ts]

'''======== Interactive Threshold Setup ========='''
def calculate_threshold(sdr, th_cycles, downsample_factor, mseq_upsampled1, mseq_upsampled2, M_up, threshold_factor_seq1, threshold_factor_seq2):
    """Manually calculate thresholds for RIS 1 and RIS 2 using correlation"""
    corr_final_first_seq = [0]
    corr_final_second_seq = [0]

    # -- RIS 1 Threshold --
    user_input = input("Initiate RIS 1 threshold process? Y/N ")
    if user_input.strip().lower() == 'y':
        print('Calculating RIS 1 threshold...')
        for _ in range(th_cycles):
            Rx = sdr.rx()[0][::downsample_factor]
            envelope = np.abs(Rx) / 2**12
            envelope -= np.mean(envelope)
            envelope /= np.max(envelope)
            corr = np.abs(correlate(mseq_upsampled1, envelope, mode='full')) / M_up
            corr_final_first_seq.append(np.max(corr))
        th_1 = threshold_factor_seq1 * np.mean(corr_final_first_seq[1:])
        print(f'TH1 = {th_1:.3f}')
    else:
        th_1 = 0
        print("RIS 1 threshold skipped or invalid. TH1 = 0")

    # -- RIS 2 Threshold --
    user_input = input("Initiate RIS 2 threshold process? Y/N ")
    if user_input.strip().lower() == 'y':
        print('Calculating RIS 2 threshold...')
        for _ in range(th_cycles):
            Rx = sdr.rx()[0][::downsample_factor]
            envelope = np.abs(Rx) / 2**12
            envelope -= np.mean(envelope)
            envelope /= np.max(envelope)
            corr = np.abs(correlate(mseq_upsampled2, envelope, mode='full')) / M_up
            corr_final_second_seq.append(np.max(corr))
        th_2 = threshold_factor_seq2 * np.mean(corr_final_second_seq[1:])
        print(f'TH2 = {th_2:.3f}')
    else:
        th_2 = 0
        print("RIS 2 threshold skipped or invalid. TH2 = 0")

    input("Press Enter to continue...")
    return th_1, th_2


'''====================== SETTINGS ======================'''

# >>> MODIFY THIS IF NEEDED (USB PORT) <<<
sdr = adi.ad9361(uri='usb:1.12.5')

# >>> MODIFY THESE THRESHOLD FACTORS IF ENVIRONMENT CHANGES <<<
threshold_factor_seq1 = 3
threshold_factor_seq2 = 2

# SDR configuration
samp_rate = 5.3e5
NumSamples = 300000
rx_lo = 5.3e9
rx_mode = "manual"
rx_gain = 0
fc0 = int(200e3)

[fs, ts] = conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain, NumSamples)

# Predefined sequences for RIS 1 and RIS 2
mseq1 = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
mseq2 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
amp = 1  # Binary symbols: 0 = +1, 1 = −1

# Upsample for better resolution in correlation
mseq1 = np.where(mseq1 == 0, amp, -amp)
mseq2 = np.where(mseq2 == 0, amp, -amp)
downsample_factor = 180
sps = 18750 / downsample_factor
mseq_upsampled1 = np.repeat(mseq1, sps)
mseq_upsampled2 = np.repeat(mseq2, sps)
M_up = len(mseq1) * sps

# Flush SDR buffers
for _ in range(5):
    sdr.rx()

'''====================== PLOT SETUP ======================'''
plt.ion()
fig = plt.figure()
bx = fig.add_subplot(111)
line1, = bx.plot([], [], label="RIS #1", marker='*')
line2, = bx.plot([], [], label="RIS #2")
line3, = bx.plot([], [], label="TH1")
line4, = bx.plot([], [], label="TH2")
bx.legend()
bx.set_title("RIS Detection and Identification")
bx.set_xlabel("Time Index")
bx.set_ylabel("Correlation Amplitude")

'''====================== THRESHOLD ESTIMATION ======================'''
th_cycles = 10
th_1, th_2 = calculate_threshold(
    sdr, th_cycles, downsample_factor,
    mseq_upsampled1, mseq_upsampled2,
    M_up, threshold_factor_seq1, threshold_factor_seq2
)

'''====================== MAIN LOOP ======================'''
num_reads = 10000
averaging_factor = 5
window_size = 30

t = []
corr_av_1 = []
corr_av_2 = []

visualizer = BinaryStateVisualizer()

try:
    for i in range(num_reads):
        if len(corr_av_1) > averaging_factor:
            corr_av_1 = corr_av_1[-averaging_factor:]
            corr_av_2 = corr_av_2[-averaging_factor:]

        t.append(i)
        Rx = sdr.rx()[0][::downsample_factor]
        envelope = np.abs(Rx) / 2 ** 12
        envelope -= np.mean(envelope)
        envelope /= np.max(envelope)

        corr_1 = np.max(np.abs(correlate(mseq_upsampled1, envelope, mode='full')) / M_up)
        corr_2 = np.max(np.abs(correlate(mseq_upsampled2, envelope, mode='full')) / M_up)

        corr_av_1.append(corr_1)
        corr_av_2.append(corr_2)

        # Determine RIS state (1 = detected, 0 = not detected)
        RIS_1_state = int(corr_av_1[-1] > th_1)
        RIS_2_state = int(corr_av_2[-1] > th_2)

        visualizer.update_states([RIS_1_state, RIS_2_state])

        # Keep window size consistent
        if i > window_size:
            t = t[-window_size:]
            corr_av_1 = corr_av_1[-window_size:]
            corr_av_2 = corr_av_2[-window_size:]

        # Update plot
        line1.set_data(t, corr_av_1)
        line2.set_data(t, corr_av_2)
        line3.set_data(t, [th_1] * len(corr_av_1))
        line4.set_data(t, [th_2] * len(corr_av_2))

        bx.relim()
        bx.autoscale_view()
        plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Measurement manually stopped.")

plt.ioff()
plt.show()

# Optional beep when done
if i > 40:
    print('\a')  # System bell
sdr.tx_destroy_buffer()
