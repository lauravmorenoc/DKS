import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
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
 
def conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,buffer_size):
    '''Configure properties for the Radio'''
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(fc0 * 3)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(buffer_size)
    sdr._rxadc.set_kernel_buffers_count(1)  # Set buffers to 1 to avoid stale data on Pluto
    fs=int(sdr.sample_rate)
    ts=1/fs
    return [fs, ts]

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


''' Variables '''

samp_rate = 5.3e5    # must be <=30.72 MHz if both channels are enabled (530000)
NumSamples = 300000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
fc0 = int(200e3)

''' Control Variables '''
threshold_factor=3
num_av_corr=5
downsample_factor=180
th_cycles=10
num_reads=10000
averaging_factor=5

'''Create Radios'''

'''sdr=adi.ad9361(uri='ip:192.168.2.1')'''
sdr=adi.ad9361(uri='usb:1.14.5') # Rx
[fs, ts]=conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,NumSamples)

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
    data = sdr.rx()

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

th=calculate_threshold(sdr,th_cycles,downsample_factor,mseq_upsampled,M_up, threshold_factor)

window_size = 30 # int(num_reads * keep_percent)
corr_av=[]
t=[]
corr_final=[0]

if __name__ == "__main__":
   visualizer = BinaryStateVisualizer()
   try:
    for i in range(num_reads):
       
        if(len(corr_av)>averaging_factor):
            corr_array=corr_array[-averaging_factor:] ### here
          
        t=np.append(t,i)
        data = sdr.rx()
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

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done