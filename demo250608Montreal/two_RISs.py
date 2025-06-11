import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
from scipy.signal import correlate
import threading
from collections import deque
import subprocess
import re

# These will be modified in real-time
threshold_factors = {
    "seq1": 3.0,
    "seq2": 2.0
}

''' Functions '''

class BinaryStateVisualizer:
    def __init__(self):
        self.detected_states = [0, 0]
        self.actual_states = [0, 0]

        self.fig, (self.ax_logo, self.ax_circles, self.ax_table) = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'width_ratios': [0.5, 2, 1.7]})

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.image as mpimg

        logo_img = mpimg.imread("logo.png")
        imagebox = OffsetImage(logo_img, zoom=0.9)  # Adjust zoom here (e.g. 0.2 for smaller)

        ab = AnnotationBbox(imagebox, (0.2, 0.8), frameon=False)  # Centered in ax_logo
        self.ax_logo.add_artist(ab)
        self.ax_logo.axis('off')

        # Circle centers: (x, y)
        centers = [(0, 1), (1, 1), (0, 0), (1, 0)]  # Row: Detected, Actual | Col: RIS 1, RIS 2
        self.circles = [plt.Circle(center, 0.3, fc='red', edgecolor='black') for center in centers]

        for circle in self.circles:
            self.ax_circles.add_patch(circle)

        # Add column headers ("RIS 1", "RIS 2")
        self.ax_circles.text(0, 1.7, "RIS 1", ha='center', va='center', fontsize=12, fontweight='bold')
        self.ax_circles.text(1, 1.7, "RIS 2", ha='center', va='center', fontsize=12, fontweight='bold')

        # Add row labels ("Detected", "Actual state")
        self.ax_circles.text(-0.6, 1, "Detected", ha='right', va='center', fontsize=12, fontweight='bold')
        self.ax_circles.text(-0.6, 0, "Actual state", ha='right', va='center', fontsize=12, fontweight='bold')

        self.ax_circles.set_xlim(-1, 2)
        self.ax_circles.set_ylim(-0.5, 2)
        self.ax_circles.set_aspect('equal')
        self.ax_circles.axis('off')

        self.table = self.ax_table.table(
            cellText=[["", "", "", ""]],
            colLabels=["", "", "", ""],
            loc='center',
            colLoc='center',
            cellLoc='center',
            bbox=[-0.25, 0.15, 0.9, 0.4]  # full table subplot
        )

        self.ax_table.text(
            0.2, 0.9, "Detection accuracy",
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            transform=self.ax_table.transAxes
        )

        self.fig.suptitle("RIS Detection and Identification", fontsize=16, fontweight='bold')
        self.ax_table.axis('off')

        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.2)
        plt.ion()
        plt.show()

    def update_states(self, detected_states, actual_states, acc_values):
        self.detected_states = detected_states
        self.actual_states = actual_states

        for i in range(2):
            self.circles[i].set_facecolor('green' if self.detected_states[i] else 'red')     # top row
            self.circles[i + 2].set_facecolor('green' if self.actual_states[i] else 'red')   # bottom row

        # ⬇️ Unpack accuracy values
        acc_last_1, acc_last_2, acc_last_overall, acc_hist_1, acc_hist_2, acc_hist_overall = acc_values

        # ⬇️ Clear old table and recreate
        self.table._cells.clear()

        headers = ["", "RIS 1", "RIS 2"]
        rows = [
            ["Instantaneous", f"{acc_last_1}%", f"{acc_last_2}%"],
            ["Average", f"{acc_hist_1}%", f"{acc_hist_2}%"]
        ]

        for col_idx, text in enumerate(headers):
            self.table.add_cell(-1, col_idx, width=0.4, height=0.17, text=text, loc='center', facecolor='#cccccc')
        
        # Bold header text
        for col_idx in range(len(headers)):
            self.table[-1, col_idx].get_text().set_fontweight('bold')

        for row_idx, row in enumerate(rows):
            for col_idx, text in enumerate(row):
                self.table.add_cell(row_idx, col_idx, width=0.4, height=0.17, text=text, loc='center', facecolor='white')

        self.table.auto_set_font_size(False)
        self.table.set_fontsize(12)
        self.table.scale(1.2, 1.5)

        self.fig.canvas.draw()
        plt.pause(0.1)

def calculate_threshold(sdr,th_cycles,downsample_factor,mseq_upsampled1, mseq_upsampled2,M_up):
    corr_array_first_seq=[0]
    corr_array_second_seq=[0]
    corr_final_first_seq=[0]
    corr_final_second_seq=[0]

    user_input = input("Initiate RIS 1 threshold process? Y/N ")
    print(f"Answered: {user_input}")
    if ((user_input=='Y')or(user_input=='y')):
        print('Calculating RIS 1 threshold')
        for j in range(th_cycles):
            data = sdr.rx()
            Rx = data[0]
            Rx=Rx[::downsample_factor]
            envelope=np.abs(Rx)/2**12
            env_mean=np.mean(envelope)
            envelope-=env_mean
            envelope=envelope/np.max(envelope)
            corr_array_first_seq=np.abs(correlate(mseq_upsampled1, envelope, mode='full'))/M_up # normalized
            corr_final_first_seq=np.append(corr_final_first_seq, np.max(corr_array_first_seq))
        th_1=np.mean(corr_final_first_seq[1:])
        print('Threshold for RIS 1 found. TH1= ')
        print(th_1)
    elif ((user_input=='N')or(user_input=='n')):
        print('Processed cancelled. Threshold for RIS 1 set to 0. Moving on.')
        th_1=0
    else:
        print('Invalid answer. Threshold for RIS 1 set to 0. Moving on.')
        th_1=0

    user_input = input("Initiate RIS 2 threshold process? Y/N ")
    print(f"Answered: {user_input}")
    if ((user_input=='Y')or(user_input=='y')):
        print('Calculating RIS 2 threshold')
        for j in range(th_cycles):
            data = sdr.rx()
            Rx = data[0]
            Rx=Rx[::downsample_factor]
            envelope=np.abs(Rx)/2**12
            env_mean=np.mean(envelope)
            envelope-=env_mean
            envelope=envelope/np.max(envelope)
            corr_array_second_seq=np.abs(correlate(mseq_upsampled2, envelope, mode='full'))/M_up # normalized
            corr_final_second_seq=np.append(corr_final_second_seq, np.max(corr_array_second_seq))
        th_2=np.mean(corr_final_second_seq[1:])
        print('Threshold for RIS 2 found. TH2= ')
        print(th_2)
    elif ((user_input=='N')or(user_input=='n')):
        print('Processed cancelled. Threshold for RIS 2 set to 0. Moving on.')
        th_2=0
    else:
        print('Invalid answer. Threshold for RIS 2 set to 0. Moving on.')
        th_2=0
    
    user_input = input("Press any key and then enter to continue running code")
    print("Moving on")

    return th_1, th_2

def read_actual_state(filepath="output.txt"):
    try:
        with open(filepath, "r") as file:
            last_line = file.readlines()[-1].strip()
            if len(last_line) == 2 and all(c in "01" for c in last_line):
                return [int(last_line[0]), int(last_line[1])]
            else:
                return [0, 0]  # Fallback if line is malformed
    except Exception as e:
        print(f"Warning: Could not read actual state from file: {e}")
        return [0, 0]

def threshold_input_listener():
    print("Threshold factor adjustment thread started.")
    print("Enter new threshold factors as two numbers separated by space. Example: 3.2 1.9")
    while True:
        try:
            user_input = input()
            values = user_input.strip().split()
            if len(values) != 2:
                raise ValueError("Please enter exactly two values.")
            val1, val2 = float(values[0]), float(values[1])
            threshold_factors["seq1"] = max(0, val1)
            threshold_factors["seq2"] = max(0, val2)
            print(f"Updated threshold factors → seq1: {val1}, seq2: {val2}")
        except Exception as e:
            print(f"Invalid input. Use format like '3.5 2.1'. Error: {e}")

def find_sdr_uris():
    try:
        import subprocess
        import re

        result = subprocess.run(["iio_info", "-s"], capture_output=True, text=True)
        output = result.stdout

        uri_tx = None
        uri_rx = None

        # Match lines like:
        # serial=104473... [usb:1.10.5]
        pattern = re.compile(r'serial=(\w+).*?\[(usb:[\d\.]+)\]')

        for match in pattern.finditer(output):
            serial = match.group(1)
            uri = match.group(2)

            print(f"Detected device → Serial: {serial}, URI: {uri}")

            if serial.endswith("51"):
                uri_tx = uri
                print("Assigned as Tx")
            elif serial.endswith("74"):
                uri_rx = uri
                print("Assigned as Rx")

        if not uri_tx or not uri_rx:
            raise RuntimeError(f"Could not find both required SDRs. Found TX={uri_tx}, RX={uri_rx}")

        return uri_tx, uri_rx

    except Exception as e:
        print(f"Error during SDR URI detection: {e}")
        raise

def compute_accuracy(lst):
    return round(100 * np.mean(lst), 1) if lst else 0

''' Variables '''

samp_rate = 5.3e5    # must be <=30.72 MHz if both channels are enabled (530000)
ts=1/samp_rate
NumSamples = 300000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
tx_gain = 0
fc0 = int(200e3)

''' Control Variables '''
threshold_factor_seq1=3
threshold_factor_seq2=2
num_av_corr=5
downsample_factor=180
th_cycles=10
num_reads=10000
averaging_factor=5

'''Create Radios'''
uri_tx, uri_rx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)
sdr_rx = adi.ad9361(uri=uri_rx)
#sdr_tx=sdr_rx=adi.ad9361(uri='usb:1.5.5')
sdr_tx.sample_rate = sdr_rx.sample_rate = int(samp_rate)
sdr_tx.rx_rf_bandwidth = sdr_rx.rx_rf_bandwidth = int(3 * fc0)
sdr_tx.rx_lo = sdr_rx.rx_lo = int(rx_lo)
sdr_tx.gain_control_mode = sdr_rx.gain_control_mode = "manual"
sdr_tx.rx_hardwaregain_chan0 = sdr_rx.rx_hardwaregain_chan0 = rx_gain
sdr_tx.rx_buffer_size = sdr_rx.rx_buffer_size = NumSamples
sdr_tx.tx_rf_bandwidth = int(3 * fc0)
sdr_tx.tx_lo = int(rx_lo)
sdr_tx.tx_cyclic_buffer = True
sdr_tx.tx_hardwaregain_chan0 = tx_gain
sdr_tx.tx_buffer_size = int(2**18)

# Transmit constant BPSK pattern (all 0s)
num_symbols = 10
sps = 16
x_symbols = np.ones(num_symbols)
samples = np.repeat(x_symbols, sps)
samples_tx = samples * (2**14)
sdr_tx.tx([samples_tx, samples_tx])

''' Pre-designed sequences '''
mseq1=np.array([0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0])
mseq2=np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])
#mseq2=np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
M=len(mseq1)
amp=1 # signal amplitude
mseq1= np.where(mseq1 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
mseq2= np.where(mseq2 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
sps=18750/downsample_factor
mseq_upsampled1=np.repeat(mseq1, sps) # upsample sequence
mseq_upsampled2=np.repeat(mseq2, sps) # upsample sequence
M_up = M*sps

'''Collect data'''

for r in range(5):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr_rx.rx()

pause=0.00001
scanning_ts=pause+NumSamples*ts
plt.ion()
fig=plt.figure()
bx=fig.add_subplot(111)
line1, = bx.plot([],[],label="RIS #1",marker='*')
line2, = bx.plot([],[],label="RIS #2")
line3,=bx.plot([],[],label="TH1")
line4,=bx.plot([],[],label="TH2")
bx.legend()
bx.set_title("RIS Detection and Identification")
bx.set_xlabel("Time Index")
bx.set_ylabel("Correlation Amplitude")
s=0

''' Finding threshold '''

corr_array_first_seq=[0]
corr_array_second_seq=[0]
corr_final_first_seq=[0]
corr_final_second_seq=[0]

th_1_base,th_2_base=calculate_threshold(sdr_rx,th_cycles,downsample_factor,mseq_upsampled1, mseq_upsampled2,M_up)
        

window_size = 30 # int(num_reads * keep_percent)
corr_av_1=[]
corr_av_2=[]
t=[]
corr_final_first_seq=[0]
corr_final_second_seq=[0]

N = 20  # last N measurements accuracy window
lastN_RIS1, lastN_RIS2 = [], []
history_RIS1, history_RIS2 = [], []

if __name__ == "__main__":
   visualizer = BinaryStateVisualizer()
   try:
    input_thread = threading.Thread(target=threshold_input_listener, daemon=True)
    input_thread.start()
    for i in range(num_reads):
       
        if(len(corr_av_1)>averaging_factor):
            corr_array_first_seq=corr_array_first_seq[-averaging_factor:]
            corr_array_second_seq=corr_array_second_seq[-averaging_factor:]
          
        t=np.append(t,i)
        data = sdr_rx.rx()
        Rx = data[0]
        Rx=Rx[::downsample_factor]
        envelope=np.abs(Rx)/2**12
        env_mean=np.mean(envelope)
        envelope-=env_mean
        envelope=envelope/np.max(envelope)
        corr_array_first_seq=np.append(corr_array_first_seq,np.max(np.abs(correlate(mseq_upsampled1, envelope, mode='full'))/M_up)) # saves all the peaks for seq 1 correlation
        corr_array_second_seq=np.append(corr_array_second_seq,np.max(np.abs(correlate(mseq_upsampled2, envelope, mode='full'))/M_up)) # saves all the peaks for seq 2 correlation

        corr_av_1=np.append(corr_av_1, np.mean(corr_array_first_seq))
        corr_av_2=np.append(corr_av_2, np.mean(corr_array_second_seq))

        # Compute dynamic thresholds
        th_1 = threshold_factors["seq1"] * th_1_base
        th_2 = threshold_factors["seq2"] * th_2_base

        RIS_1_state = int(corr_av_1[-1] > th_1)
        RIS_2_state = int(corr_av_2[-1] > th_2)

        actual_states = read_actual_state()
        # Store match results
        correct_RIS1 = int(RIS_1_state == actual_states[0])
        correct_RIS2 = int(RIS_2_state == actual_states[1])

        # Store history
        history_RIS1.append(correct_RIS1)
        history_RIS2.append(correct_RIS2)

        # Keep rolling window for recent
        lastN_RIS1.append(correct_RIS1)
        lastN_RIS2.append(correct_RIS2)

        if len(lastN_RIS1) > N:
            lastN_RIS1.pop(0)
            lastN_RIS2.pop(0)

        acc_last_1 = compute_accuracy(lastN_RIS1)
        acc_last_2 = compute_accuracy(lastN_RIS2)
        acc_hist_1 = compute_accuracy(history_RIS1)
        acc_hist_2 = compute_accuracy(history_RIS2)

        acc_last_overall = round((acc_last_1 + acc_last_2) / 2, 1)
        acc_hist_overall = round((acc_hist_1 + acc_hist_2) / 2, 1)

        visualizer.update_states((int(RIS_1_state), int(RIS_2_state)),actual_states,(acc_last_1, acc_last_2, acc_last_overall, acc_hist_1, acc_hist_2, acc_hist_overall))
            
        if i>window_size:
            t=t[-window_size:]
            corr_av_1=corr_av_1[-window_size:]
            corr_av_2=corr_av_2[-window_size:]


        line1.set_data(t,corr_av_1)
        line2.set_data(t,corr_av_2)
        line3.set_data(t,[th_1]*len(corr_av_1))
        line4.set_data(t,[th_2]*len(corr_av_2))
        bx.set_title(f"Correlation peaks over time \n Threshold factors - 1: {threshold_factors['seq1']:.0f}, 2: {threshold_factors['seq2']:.0f}")
        bx.relim()
        bx.autoscale_view()

        plt.draw()
        plt.pause(0.01)
   except KeyboardInterrupt:
    pass
plt.ioff()
plt.show()

sdr_tx.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done
