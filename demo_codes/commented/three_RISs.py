import adi  # Pluto SDR interface
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
import time

'''======================= USER CONFIGURATION ======================='''

# >>> MODIFY THESE VALUES ACCORDING TO ENVIRONMENT <<<
threshold_factor_seq1 = 4  # Multiplier for RIS 1 detection threshold
threshold_factor_seq2 = 3  # Multiplier for RIS 2 detection threshold
threshold_factor_seq3 = 3  # Multiplier for RIS 3 detection threshold

# >>> MODIFY THIS LINE TO USE THE CORRECT USB PORT <<<
sdr = adi.ad9361(uri='usb:1.14.5')  # SDR USB address (check with Pluto GUI)

'''======================= VISUALIZATION ======================='''

class BinaryStateVisualizer:
    def __init__(self):
        self.states = [0, 0, 0]  # Initial states of the 3 RIS

        self.fig, self.ax = plt.subplots()
        centers = [(-0.3, -0.5), (1.3, -0.5), (1.3, 1)]
        labels = ["RIS 1", "RIS 2", "RIS 3"]

        self.circles = [plt.Circle(center, 0.4, fc='red', edgecolor='black') for center in centers]
        for circle in self.circles:
            self.ax.add_patch(circle)
        for (x, y), label in zip(centers, labels):
            self.ax.text(x, y + 0.6, label, ha='center', va='center', fontsize=12, fontweight='bold')

        self.ax.set_xlim(-1, 3)
        self.ax.set_ylim(-1.5, 3)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        plt.ion()
        plt.show()

    def update_states(self, new_states):
        """Update RIS circles based on new binary states."""
        if len(new_states) == 3:
            self.states = new_states
            for i, circle in enumerate(self.circles):
                circle.set_facecolor('green' if self.states[i] else 'red')
            self.fig.canvas.draw()
            plt.pause(0.1)

'''======================= SDR CONFIGURATION ======================='''

def conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain, buffer_size):
    """Configure the SDR receiver."""
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(fc0 * 3)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(buffer_size)
    sdr._rxadc.set_kernel_buffers_count(1)
    fs = int(sdr.sample_rate)
    ts = 1 / fs
    return fs, ts

'''======================= THRESHOLD CALCULATION ======================='''

def calculate_threshold(sdr, th_cycles, downsample_factor,
                        mseq1, mseq2, mseq3, M_up,
                        factor1, factor2, factor3):
    """Interactive threshold determination for each RIS."""
    def process(seq, factor, label):
        user_input = input(f"Initiate {label} threshold process? Y/N ")
        if user_input.lower() == 'y':
            print(f'Calculating {label} threshold...')
            max_corrs = []
            for _ in range(th_cycles):
                Rx = sdr.rx()[0][::downsample_factor]
                envelope = np.abs(Rx) / 2**12
                envelope -= np.mean(envelope)
                envelope /= np.max(envelope)
                corr = np.abs(correlate(seq, envelope, mode='full')) / M_up
                max_corrs.append(np.max(corr))
            threshold = factor * np.mean(max_corrs)
            print(f'Threshold for {label} found: {threshold:.4f}')
            return threshold
        else:
            print(f"{label} threshold skipped or invalid. Set to 0.")
            return 0

    th1 = process(mseq1, factor1, "RIS 1")
    th2 = process(mseq2, factor2, "RIS 2")
    th3 = process(mseq3, factor3, "RIS 3")

    input("Press Enter to continue...")
    return th1, th2, th3

'''======================= PARAMETERS ======================='''

samp_rate = 5.3e5        # Must be <= 30.72 MHz if both channels are active
NumSamples = 300000      # SDR buffer size
rx_lo = 5.3e9            # Center frequency (Hz)
rx_mode = "manual"       # Gain control: "manual" or "slow_attack"
rx_gain = 0              # Receiver gain (dB)
fc0 = int(200e3)

fs, ts = conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain, NumSamples)

# Threshold and sampling settings
num_reads = 10000
th_cycles = 10
downsample_factor = 180
averaging_factor = 5
window_size = 30

'''======================= SEQUENCES ======================='''

# Binary test sequences for RIS 1, 2, and 3
mseq1 = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0])
mseq2 = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
mseq3 = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0])

# Convert to bipolar: 0 → +1, 1 → –1
mseq1 = np.where(mseq1 == 0, 1, -1)
mseq2 = np.where(mseq2 == 0, 1, -1)
mseq3 = np.where(mseq3 == 0, 1, -1)

sps = int(18750 / downsample_factor)  # Samples per symbol after upsampling
mseq_upsampled1 = np.repeat(mseq1, sps)
mseq_upsampled2 = np.repeat(mseq2, sps)
mseq_upsampled3 = np.repeat(mseq3, sps)
M_up = len(mseq1) * sps

# Flush old SDR data
for _ in range(5):
    sdr.rx()

'''======================= PLOT SETUP ======================='''

plt.ion()
fig = plt.figure()
bx = fig.add_subplot(111)
line1, = bx.plot([], [], label="RIS #1", marker='*')
line2, = bx.plot([], [], label="RIS #2", marker='+')
line3, = bx.plot([], [], label="RIS #3", marker='x')
line4, = bx.plot([], [], label="TH1")
line5, = bx.plot([], [], label="TH2")
line6, = bx.plot([], [], label="TH3")
bx.legend()
bx.set_title("RIS Detection and Identification")
bx.set_xlabel("Time Index")
bx.set_ylabel("Correlation Amplitude")

'''======================= FIND THRESHOLDS ======================='''

th_1, th_2, th_3 = calculate_threshold(
    sdr, th_cycles, downsample_factor,
    mseq_upsampled1, mseq_upsampled2, mseq_upsampled3,
    M_up, threshold_factor_seq1, threshold_factor_seq2, threshold_factor_seq3
)

'''======================= MAIN LOOP ======================='''

t = []
corr_av_1 = []
corr_av_2 = []
corr_av_3 = []

corr_array_first_seq = [0]
corr_array_second_seq = [0]
corr_array_third_seq = [0]

visualizer = BinaryStateVisualizer()

try:
    for i in range(num_reads):
        if len(corr_av_1) > averaging_factor:
            corr_array_first_seq = corr_array_first_seq[-averaging_factor:]
            corr_array_second_seq = corr_array_second_seq[-averaging_factor:]
            corr_array_third_seq = corr_array_third_seq[-averaging_factor:]

        t.append(i)
        Rx = sdr.rx()[0][::downsample_factor]
        envelope = np.abs(Rx) / 2 ** 12
        envelope -= np.mean(envelope)
        envelope /= np.max(envelope)

        corr_array_first_seq.append(np.max(np.abs(correlate(mseq_upsampled1, envelope, mode='full')) / M_up))
        corr_array_second_seq.append(np.max(np.abs(correlate(mseq_upsampled2, envelope, mode='full')) / M_up))
        corr_array_third_seq.append(np.max(np.abs(correlate(mseq_upsampled3, envelope, mode='full')) / M_up))

        corr_av_1.append(np.mean(corr_array_first_seq))
        corr_av_2.append(np.mean(corr_array_second_seq))
        corr_av_3.append(np.mean(corr_array_third_seq))

        RIS_state = [
            int(corr_av_1[-1] > th_1),
            int(corr_av_2[-1] > th_2),
            int(corr_av_3[-1] > th_3)
        ]

        visualizer.update_states(RIS_state)

        # Keep window size fixed
        if i > window_size:
            t = t[-window_size:]
            corr_av_1 = corr_av_1[-window_size:]
            corr_av_2 = corr_av_2[-window_size:]
            corr_av_3 = corr_av_3[-window_size:]

        # Update plot
        line1.set_data(t, corr_av_1)
        line2.set_data(t, corr_av_2)
        line3.set_data(t, corr_av_3)
        line4.set_data(t, [th_1] * len(corr_av_1))
        line5.set_data(t, [th_2] * len(corr_av_2))
        line6.set_data(t, [th_3] * len(corr_av_3))

        bx.relim()
        bx.autoscale_view()
        plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Interrupted by user.")

plt.ioff()
plt.show()

# Clean up
sdr.tx_destroy_buffer()
if i > 40:
    print('\a')  # Beep when finished
