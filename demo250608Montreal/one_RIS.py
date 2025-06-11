import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
import threading


''' Functions '''

class BinaryStateVisualizer:
    def __init__(self):
        self.state = 0  # Detected state
        self.actual_state_text = None
        self.acc_table_text = None

        self.fig, self.ax = plt.subplots(figsize=(4, 2.5))
        center = (0.5, 0)
        label = "RIS"
        self.circle = plt.Circle(center, 0.3, fc='red', edgecolor='black')
        self.ax.add_patch(self.circle)
        self.ax.text(center[0], center[1] + 0.6, label, ha='center', va='center',
                     fontsize=12, fontweight='bold')

        self.actual_state_text = self.ax.text(1.1, 0.4, "", ha='left', va='center',
                                              fontsize=10, bbox=dict(boxstyle="round", facecolor='white'))

        self.acc_table_text = self.ax.text(1.1, -0.3, "", ha='left', va='top',
                                           fontsize=10, family='monospace', bbox=dict(boxstyle="round", facecolor='lightgrey'))

        self.ax.set_xlim(-0.5, 2)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        plt.ion()
        plt.show()

    def update_state(self, new_state, actual_bit=None, acc_13=None, acc_hist=None):
        if new_state in [0, 1]:
            self.state = new_state
            self.circle.set_facecolor('green' if self.state else 'red')

        if actual_bit is not None:
            state_label = "Actual: ON" if actual_bit else "Actual: OFF"
            self.actual_state_text.set_text(state_label)
            self.actual_state_text.set_color('green' if actual_bit else 'red')

        if acc_13 is not None and acc_hist is not None:
            acc_text = f"{'Accuracy':<12}\n{'Last 13:':<12}{acc_13:.0f}%\n{'Historical:':<12}{acc_hist:.0f}%"
            self.acc_table_text.set_text(acc_text)

        self.fig.canvas.draw()
        plt.pause(0.01)
        
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

def listen_for_input():
    global user_input_value
    while True:
        try:
            value = input("Enter threshold factor: ")
            user_input_value = value.strip()
        except:
            pass

''' Variables '''

samp_rate = 5.3e5    # must be <=30.72 MHz if both channels are enabled (530000)
NumSamples = 300000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
tx_gain = 0
fc0 = int(200e3)
user_input_value = None  # Shared variable to store input

''' Control Variables '''
threshold_factor=3
num_av_corr=5
downsample_factor=180
th_cycles=10
num_reads=10000
averaging_factor=5

'''Create Radios'''
sdr_tx = adi.ad9361(uri='usb:1.57.5')
sdr_rx = adi.ad9361(uri='usb:1.55.5')
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
mseq=np.array([0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0])
M=len(mseq)
amp=1 # signal amplitude
mseq= np.where(mseq == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
sps=18750/downsample_factor
mseq_upsampled=np.repeat(mseq, sps) # upsample sequence
M_up = M*sps

'''Collect data'''

for r in range(5):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr_rx.rx()

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


''' Finding threshold '''

corr_array=[0]
corr_final=[0]

base_threshold=calculate_threshold(sdr_rx,th_cycles,downsample_factor,mseq_upsampled,M_up, threshold_factor)

window_size = 30 # int(num_reads * keep_percent)
corr_av=[]
t=[]

# Probability
rolling_window = 13  # or set this to any number of recent measurements
actual_states = []
detected_states = []


if __name__ == "__main__":
   visualizer = BinaryStateVisualizer()
   try:
    listener_thread = threading.Thread(target=listen_for_input, daemon=True)
    listener_thread.start()

    # Historical counters
    correct_total = 0
    total_total = 0

    for i in range(num_reads):

        # If user entered a value, try updating the threshold_factor
        if user_input_value is not None:
            try:
                threshold_factor = float(user_input_value)
                print(f"New threshold_factor: {threshold_factor}")
            except ValueError:
                print(f"Ignoring invalid input: {user_input_value}")
            user_input_value = None  # reset

        th = threshold_factor * base_threshold

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

        # Read actual RIS state from file (bit 1)
        try:
            with open("output.txt", "r") as f:
                bit_string = f.read().strip()
            actual_bit = int(bit_string[1])
        except:
            actual_bit = None  # If failed to read, don't update actual state

        # Append new data
        if actual_bit is not None:
            actual_states.append(actual_bit)
            detected_states.append(int(RIS_state))

            # Keep only the last N elements
            if len(actual_states) > rolling_window:
                actual_states = actual_states[-rolling_window:]
                detected_states = detected_states[-rolling_window:]

            # Compute detection accuracy
            correct = sum([1 for a, d in zip(actual_states, detected_states) if a == d])
            total = len(actual_states)
            accuracy = correct / total if total > 0 else 0
            # Update historical stats
            correct_total += int(actual_bit == int(RIS_state))
            total_total += 1

            # Show in plot title
            bx.set_title(f"RIS Detection and Identification\n Threshold factor: {threshold_factor}")
            visualizer.update_state(int(RIS_state), actual_bit, acc_13=accuracy*100, acc_hist=(correct_total/total_total)*100 if total_total > 0 else 0)


        if i>window_size:
            t=t[-window_size:]
            corr_av=corr_av[-window_size:]

        line1.set_data(t,corr_av)
        line2.set_data(t,[th]*len(corr_av))

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