import serial
import time
import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



# ===================== USER PARAMETERS =====================
ris_port = 'COM18'
baudrate = 115200
reads_per_check = 3
group_size = 16
change_period= 1 # seconds
num_rows = 16
num_cols = 16
rx_lo = 5.3e9
sample_rate = 5.3e5
NumSamples = 300000
rx_gain = 0
tx_gain = 0
# ===========================================================

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


def ris_init(port, baudrate):
    ris = serial.Serial(port, baudrate, timeout=1)
    ris.write(b'!Reset\n')
    time.sleep(1)
    while ris.in_waiting > 0:
        response = ris.readline().decode().strip()
        print(f"Reset Response: {response}")
        time.sleep(0.1)
    return ris


def generate_pattern(state_array):
    bitstring = ''.join(['1' if x else '0' for x in state_array])
    hexstring = hex(int(bitstring, 2))[2:].zfill(64)
    return '!0x' + hexstring


def send_pattern(ris, pattern):
    ris.write((pattern + '\n').encode())
    time.sleep(0.05)
    if ris.in_waiting > 0:
        return ris.readline().decode().strip()
    return None


def measure_power(sdr_rx, NumSamples, reads_per_check):
    powers = []
    for _ in range(reads_per_check):
        data = sdr_rx.rx()
        Rx_0 = data[0]
        y = Rx_0 * np.hamming(NumSamples)
        sp = np.abs(np.fft.fftshift(np.fft.fft(y)[1:-1]))
        mag = np.abs(sp) / (np.sum(np.hamming(NumSamples)) / 2)
        dbfs = 20 * np.log10(mag / (2**12))
        peak_power = np.max(dbfs)
        powers.append(peak_power)
    return np.mean(powers)


# ---- MAIN SCRIPT ----
ris = ris_init(ris_port, baudrate)
state = [0] * (num_rows * num_cols)  # initial state: all OFF
send_pattern(ris, generate_pattern(state))

# Configure SDR
uri_tx, uri_rx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)
sdr_rx = adi.ad9361(uri=uri_rx)
sdr_tx.sample_rate = sdr_rx.sample_rate = int(sample_rate)
sdr_tx.rx_rf_bandwidth = sdr_rx.rx_rf_bandwidth = int(3 * 0)
sdr_tx.rx_lo = sdr_rx.rx_lo = int(rx_lo)
sdr_tx.gain_control_mode = sdr_rx.gain_control_mode = "manual"
sdr_tx.rx_hardwaregain_chan0 = sdr_rx.rx_hardwaregain_chan0 = rx_gain
sdr_tx.rx_buffer_size = sdr_rx.rx_buffer_size = NumSamples
sdr_tx.tx_rf_bandwidth = int(3 * 0)
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

# Compute group top-left positions
group_side = int(np.sqrt(group_size))
power_history = []

# Initial power
current_power = measure_power(sdr_rx, NumSamples, reads_per_check)
power_history.append(current_power)

user_input = input("Initiate optimization process? Y/N ")
if ((user_input=='Y')or(user_input=='y')):
    print('Optimizing')
    # ============================= MAXIMIZING =============================

    for row in range(0, num_rows, group_side):
        for col in range(0, num_cols, group_side):
            print(f"\nChecking group at ({row}, {col})")

            # Get indices of this 4x4 group
            indices = []
            for r in range(group_side):
                for c in range(group_side):
                    i = (row + r) * num_cols + (col + c)
                    indices.append(i)

            # Flip group bits
            for idx in indices:
                state[idx] ^= 1  # toggle

            # Send new pattern and measure power
            send_pattern(ris, generate_pattern(state))
            new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

            print(f"Old power: {current_power:.2f} dBFS, New power: {new_power:.2f} dBFS")

            if new_power >= current_power:
                print("→ Change accepted.")
                current_power = new_power
            else:
                print("→ Change reverted.")
                # revert changes
                for idx in indices:
                    state[idx] ^= 1
                send_pattern(ris, generate_pattern(state))

            power_history.append(current_power)
            time.sleep(change_period)


    # Save or plot power history
    plt.figure()
    plt.plot(power_history, marker='o')
    plt.xlabel("Group index")
    plt.ylabel("Max Power (dBFS)")
    plt.title("Power evolution during RIS optimization")
    plt.grid()
    plt.show(block=False)

    # Saving in text file
    hex_pattern = generate_pattern(state)
    with open("optimized_max_ris_pattern_hex.txt", "w") as f:
        f.write(hex_pattern)



    # ============================= MINIMIZING =============================
    print('Minimizing now')
    power_history = []

    # Initial power
    current_power = measure_power(sdr_rx, NumSamples, reads_per_check)
    power_history.append(current_power)

    # Get min power pattern
    for row in range(0, num_rows, group_side):
        for col in range(0, num_cols, group_side):
            print(f"\nChecking group at ({row}, {col})")

            # Get indices of this 4x4 group
            indices = []
            for r in range(group_side):
                for c in range(group_side):
                    i = (row + r) * num_cols + (col + c)
                    indices.append(i)

            # Flip group bits
            for idx in indices:
                state[idx] ^= 1  # toggle

            # Send new pattern and measure power
            send_pattern(ris, generate_pattern(state))
            new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

            print(f"Old power: {current_power:.2f} dBFS, New power: {new_power:.2f} dBFS")

            if new_power <= current_power:
                print("→ Change accepted.")
                current_power = new_power
            else:
                print("→ Change reverted.")
                # revert changes
                for idx in indices:
                    state[idx] ^= 1
                send_pattern(ris, generate_pattern(state))

            power_history.append(current_power)
            time.sleep(change_period)


    # Save or plot power history
    plt.figure()
    plt.plot(power_history, marker='o')
    plt.xlabel("Group index")
    plt.ylabel("Max Power (dBFS)")
    plt.title("Power evolution during RIS optimization")
    plt.grid()
    plt.show()

    # Saving in text file
    hex_pattern = generate_pattern(state)
    with open("optimized_min_ris_pattern_hex.txt", "w") as f:
        f.write(hex_pattern)

    ris.close()
else:
    print('Moving on')


# ================================= RUNNING CORRELATION PROCESS =================================

from scipy.signal import correlate


''' Functions '''

class BinaryStateVisualizer:
    def __init__(self):
        self.state = 0  # Initial state

        # Set up the figure
        self.fig, self.ax = plt.subplots(figsize=(2, 2))
        center = (0.5, 0)
        label = "RIS"
        self.circle = plt.Circle(center, 0.3, fc='red', edgecolor='black')
        self.ax.add_patch(self.circle)

        self.ax.text(center[0], center[1] + 0.6, label, ha='center', va='center', fontsize=12, fontweight='bold')

        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        plt.ion()  # Turn on interactive mode
        plt.show()

    def update_state(self, new_state):
        """Update the binary state (list of 4 elements: 0 or 1)."""
        if new_state in [0, 1]:
            self.state = new_state
            self.circle.set_facecolor('green' if self.state else 'red')
            self.fig.canvas.draw()
            plt.pause(0.1)


def calculate_threshold(sdr,th_cycles,downsample_factor,mseq_upsampled,M_up, threshold_factor):
    corr_array=[0]
    corr_final=[0]
    user_input = input("Initiate threshold process? Y/N ")
    print(f"Answered: {user_input}")
    if ((user_input=='Y')or(user_input=='y')):
        print('Calculating threshold')
        for j in range(th_cycles):
            data = sdr.rx()
            Rx = data[0]
            Rx=Rx[::downsample_factor]
            envelope=np.abs(Rx)/2**12
            env_mean=np.mean(envelope)
            envelope-=env_mean
            envelope=envelope/np.max(envelope)
            corr_array=np.abs(correlate(mseq_upsampled, envelope, mode='full'))/M_up # normalized
            corr_final=np.append(corr_final, np.max(corr_array))
        th=threshold_factor*np.mean(corr_final[1:])
        print('Threshold found. TH= ')
        print(th)
    elif ((user_input=='N')or(user_input=='n')):
        print('Processed cancelled. Threshold set to 0. Moving on.')
        th=0
    else:
        print('Invalid answer. Threshold set to 0. Moving on.')
        th=0
    
    user_input = input("Press enter to continue")
    print("Moving on")

    return th


''' Control Variables '''
threshold_factor=3
num_av_corr=5
downsample_factor=180
th_cycles=10
num_reads=69
averaging_factor=5
fs=int(sdr_rx.sample_rate)
ts=1/fs

''' Pre-designed sequences '''
#mseq=np.array([0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0])
mseq=np.array([0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0])
M=len(mseq)
amp=1 # signal amplitude
mseq= np.where(mseq == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
sps=18750/downsample_factor
mseq_upsampled=np.repeat(mseq, sps) # upsample sequence
M_up = M*sps

'''Collect data'''

for r in range(5):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr_rx.rx()

pause=0.00001
scanning_ts=pause+NumSamples*ts
plt.ion()
fig=plt.figure()
bx=fig.add_subplot(111)
line1, = bx.plot([],[],label="Corr. peaks",marker='*')
line2,=bx.plot([],[],label="TH")
bx.legend()
bx.legend()
bx.set_title("RIS Detection and Identification")
bx.set_xlabel("Time Index")
bx.set_ylabel("Correlation Amplitude")
s=0

''' Finding threshold '''

corr_array=[0]
corr_final=[0]

th=calculate_threshold(sdr_rx,th_cycles,downsample_factor,mseq_upsampled,M_up, threshold_factor)

window_size = 30 # int(num_reads * keep_percent)
corr_av=[]
t=[]
corr_final=[0]
change=0

colors = ['blue', 'orange', 'green']
labels = ['Before RIS on', 'Baseline', 'Optimized']

segments_t = []
segments_corr = []


if __name__ == "__main__":
   visualizer = BinaryStateVisualizer()
   try:
    for i in range(num_reads):
       
        if(len(corr_av)>averaging_factor):
            corr_array=corr_array[-averaging_factor:] ### here
          
        t=np.append(t,i)
        data = sdr_rx.rx()
        Rx = data[0]
        Rx=Rx[::downsample_factor]
        envelope=np.abs(Rx)/2**12
        env_mean=np.mean(envelope)
        envelope-=env_mean
        envelope=envelope/np.max(envelope)
        corr_array=np.append(corr_array,np.max(np.abs(correlate(mseq_upsampled, envelope, mode='full'))/M_up)) # saves all the peaks for seq correlation

        corr_av=np.append(corr_av, np.mean(corr_array))

        RIS_state= np.abs((corr_av[-1:]>th)*1)

        visualizer.update_state((int(RIS_state)))
            
        if i>window_size:
            t=t[-window_size:]
            corr_av=corr_av[-window_size:]

        bx.clear()

        # Plot stored segments
        for k in range(len(segments_t)):
            bx.plot(segments_t[k], segments_corr[k], color=colors[k], label=labels[k], marker='*')

        # Also plot the current one in progress
        if change < 3:
            bx.plot(t, corr_av, color=colors[change], label=labels[change], marker='*')

        # Threshold line (for full t)
        all_t = np.concatenate(segments_t + [t])
        bx.plot(all_t, [th] * len(all_t), label="Threshold", color='red', linestyle='--')

        bx.set_title("RIS Detection and Identification")
        bx.set_xlabel("Time Index")
        bx.set_ylabel("Correlation Amplitude")
        bx.legend(loc="upper left")
        bx.grid(True)

        handles, labels = bx.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        bx.legend(by_label.values(), by_label.keys(), loc="upper left")

        bx.relim()
        bx.autoscale_view()

        plt.draw()
        plt.pause(0.01)

        if ((i>num_reads//3)&(change==0)):
            user_input = input("First measurements done. Turn on RIS")
            print("Moving on")
            segments_t.append(t.copy())
            segments_corr.append(corr_av.copy())
            t = []
            corr_av = []
            change=1

        if ((i>2*num_reads//3)&(change==1)):
            user_input = input("Done. Change RIS patterns")
            print("Moving on")
            segments_t.append(t.copy())
            segments_corr.append(corr_av.copy())
            t = []
            corr_av = []
            change=2

            
   except KeyboardInterrupt:
    pass
   

plt.ioff()
plt.show()

sdr_tx.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done