import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

def conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain, buffer_size):
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

def calculate_threshold(sdr, th_cycles, downsample_factor, mseq_upsampled, M_up, threshold_factor):
    corr_final = [0]
    user_input = input("Initiate threshold process? Y/N ")
    print(f"Answered: {user_input}")
    if user_input.lower() == 'y':
        print('Calculating threshold')
        for _ in range(th_cycles):
            data = sdr.rx()[0]
            Rx = data[::downsample_factor]
            envelope = np.abs(Rx) / 2 ** 12
            envelope -= np.mean(envelope)
            envelope /= np.max(envelope)
            corr = np.abs(correlate(mseq_upsampled, envelope, mode='full')) / M_up
            corr_final.append(np.max(corr))
        th = threshold_factor * np.mean(corr_final[1:])
        print(f'Threshold found: {th}')
    else:
        print('Threshold process skipped. TH = 0')
        th = 0
    input("Press Enter to continue...")
    return th

class BinaryStateVisualizer:
    def __init__(self):
        self.state = 0
        self.fig, self.ax = plt.subplots(figsize=(2, 2))
        center = (0.5, 0)
        self.circle = plt.Circle(center, 0.3, fc='red', edgecolor='black')
        self.ax.add_patch(self.circle)
        self.ax.text(center[0], center[1] + 0.6, "RIS", ha='center', va='center', fontsize=12, fontweight='bold')
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        plt.ion()
        plt.show()

    def update_state(self, new_state):
        if new_state in [0, 1]:
            self.state = new_state
            self.circle.set_facecolor('green' if self.state else 'red')
            self.fig.canvas.draw()
            plt.pause(0.1)

def run_sdr_process():
    samp_rate = 5.3e5
    NumSamples = 300000
    rx_lo = 5.3e9
    rx_mode = "manual"
    rx_gain = 0
    fc0 = int(200e3)

    threshold_factor = 3
    num_reads = 10000
    downsample_factor = 180
    th_cycles = 10
    averaging_factor = 5
    window_size = 30

    mseq = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    M = len(mseq)
    amp = 1
    mseq = np.where(mseq == 0, amp, -amp)
    sps = int(18750 / downsample_factor)
    mseq_upsampled = np.repeat(mseq, sps)
    M_up = M * sps

    sdr = adi.ad9361(uri='usb:2.2.5')
    fs, ts = conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain, NumSamples)

    for _ in range(5):
        sdr.rx()

    th = calculate_threshold(sdr, th_cycles, downsample_factor, mseq_upsampled, M_up, threshold_factor)

    pause = 0.00001
    scanning_ts = pause + NumSamples * ts

    fig = plt.figure()
    bx = fig.add_subplot(111)
    line1, = bx.plot([], [], label="Corr. peaks", marker='*')
    line2, = bx.plot([], [], label="TH")
    bx.legend()
    bx.set_title("RIS Detection and Identification")
    bx.set_xlabel("Time Index")
    bx.set_ylabel("Correlation Amplitude")

    t = []
    corr_av = []

    visualizer = BinaryStateVisualizer()

    try:
        for i in range(num_reads):
            print('entering loop')
            if len(corr_av) > averaging_factor:
                corr_av = corr_av[-averaging_factor:]

            t.append(i)
            data = sdr.rx()[0]
            Rx = data[::downsample_factor]
            envelope = np.abs(Rx) / 2 ** 12
            envelope -= np.mean(envelope)
            envelope /= np.max(envelope)
            corr = np.max(np.abs(correlate(mseq_upsampled, envelope, mode='full')) / M_up)
            corr_av.append(corr)

            ris_state = int(np.abs(corr_av[-1] > th))
            visualizer.update_state(ris_state)

            if i > window_size:
                t = t[-window_size:]
                corr_av = corr_av[-window_size:]

            line1.set_data(t, corr_av)
            line2.set_data(t, [th] * len(corr_av))
            print(len(t), ' hola ', len(corr_av))
            bx.relim()
            bx.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopped by user.")

    plt.ioff()
    plt.show()

    sdr.tx_destroy_buffer()

    if i > 40:
        print('\a')  # system beep
