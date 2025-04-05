import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
import math
import tkinter as tk
import threading
''' Functions '''
import time
import random


class BinaryStateVisualizer:
    def __init__(self):
        #self.states = [0, 0, 0, 0]  # Initial states
        self.states = [0, 0, 0]  # Initial states

        # Set up the figure
        self.fig, self.ax = plt.subplots()
        #centers = [(-0.3,-0.5), (-0.3,1), (1.3,-0.5), (1.3,1)]
        centers = [(-0.3,-0.5),(1.3,-0.5),(1.3,1)]
        #labels = ["RIS 1 detected", "RIS 1", "RIS 2 detected", "RIS 2"]
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

        plt.ion()  # Turn on interactive mode
        plt.show()
        

    def update_states(self, new_states):
        """Update the binary states (list of 4 elements: 0 or 1)."""
        #if len(new_states) == 4:
        if len(new_states) == 3:
            self.states = new_states
            for i, circle in enumerate(self.circles):
                circle.set_facecolor('green' if self.states[i] else 'red')  # Change color
            self.fig.canvas.draw()  # Ensure the figure updates
            plt.pause(0.1)  # Small pause to allow GUI refresh
 
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

def calculate_threshold(sdr,th_cycles,downsample_factor,mseq_upsampled1, mseq_upsampled2, mseq_upsampled3,M_up, threshold_factor_seq1, threshold_factor_seq2, threshold_factor_seq3):
    corr_array_first_seq=[0]
    corr_array_second_seq=[0]
    corr_array_third_seq=[0]
    corr_final_first_seq=[0]
    corr_final_second_seq=[0]
    corr_final_third_seq=[0]

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
        th_1=threshold_factor_seq1*np.mean(corr_final_first_seq[1:])
        print('Threshold for RIS 1 found. TH1= ')
        print(th_1)
    elif ((user_input=='N')or(user_input=='n')):
        print('Processed cancelled. Threshold for RIS 1 set to 0. Moving on.')
        th_1=0
    else:
        print('Invalid answer.Threshold for RIS 1 set to 0. Moving on.')
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
        th_2=threshold_factor_seq2*np.mean(corr_final_second_seq[1:])
        print('Threshold for RIS 2 found. TH2= ')
        print(th_2)
    elif ((user_input=='N')or(user_input=='n')):
        print('Processed cancelled. Threshold for RIS 2 set to 0. Moving on.')
        th_2=0
    else:
        print('Invalid answer. Threshold for RIS 2 set to 0. Moving on.')
        th_2=0

    user_input = input("Initiate RIS 3 threshold process? Y/N ")
    print(f"Answered: {user_input}")
    if ((user_input=='Y')or(user_input=='y')):
        print('Calculating RIS 3 threshold')
        for j in range(th_cycles):
            data = sdr.rx()
            Rx = data[0]
            Rx=Rx[::downsample_factor]
            envelope=np.abs(Rx)/2**12
            env_mean=np.mean(envelope)
            envelope-=env_mean
            envelope=envelope/np.max(envelope)
            corr_array_third_seq=np.abs(correlate(mseq_upsampled3, envelope, mode='full'))/M_up # normalized
            corr_final_third_seq=np.append(corr_final_third_seq, np.max(corr_array_third_seq))
        th_3=threshold_factor_seq3*np.mean(corr_final_third_seq[1:])
        print('Threshold for RIS 3 found. TH3= ')
        print(th_3)
    elif ((user_input=='N')or(user_input=='n')):
        print('Processed cancelled. Threshold for RIS 3 set to 0. Moving on.')
        th_2=0
    else:
        print('Invalid answer. Threshold for RIS 3 set to 0. Moving on.')
        th_2=0
    
    user_input = input("Press any key and then enter to continue running code")
    print("Moving on")

    return th_1, th_2, th_3

#visualizer = BinaryStateVisualizer()
#visualizer.update_states((1,1))
''' Variables '''

samp_rate = 5.3e5    # must be <=30.72 MHz if both channels are enabled (530000)
NumSamples = 300000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
fc0 = int(200e3)
err1=[1]
err2=[1]

''' Control Variables '''
threshold_factor_seq1=4
threshold_factor_seq2=3
threshold_factor_seq3=3
num_av_corr=5
downsample_factor=180
th_cycles=10
num_reads=10000
averaging_factor=5
'''Create Radios'''

'''sdr=adi.ad9361(uri='ip:192.168.2.1')'''
sdr=adi.ad9361(uri='usb:1.10.5')
[fs, ts]=conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,NumSamples)


''' Pre-designed sequence '''

#mseq1=np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
#mseq2=np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])
mseq1=np.array([0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0])
mseq2=np.array([1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1])
mseq3=np.array([0,0,1,1,0,1,1,1,1,0,1,1,0,1,0,0])
M=len(mseq1)
amp=1 # signal amplitude
mseq1= np.where(mseq1 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
mseq2= np.where(mseq2 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
mseq3= np.where(mseq3 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp

sps=18750/downsample_factor
mseq_upsampled1=np.repeat(mseq1, sps) # upsample sequence
mseq_upsampled2=np.repeat(mseq2, sps) # upsample sequence
mseq_upsampled3=np.repeat(mseq3, sps) # upsample sequence

M_up = M*sps

'''Collect data'''

for r in range(5):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr.rx()

plt.ion()
fig=plt.figure()
bx=fig.add_subplot(111)
line1, = bx.plot([],[],label="RIS #1",marker='*')
line2, = bx.plot([],[],label="RIS #2",marker='+')
line3, = bx.plot([],[],label="RIS #3",marker='x')
line4,=bx.plot([],[],label="TH1")
line5,=bx.plot([],[],label="TH2")
line6,=bx.plot([],[],label="TH3")
bx.legend()
bx.set_title("RIS Detection and Identification")
bx.set_xlabel("Time Index")
bx.set_ylabel("Correlation Amplitude")
s=0

''' Finding threshold '''

corr_array_first_seq=[0]
corr_array_second_seq=[0]
corr_array_third_seq=[0]


th_1,th_2,th_3=calculate_threshold(sdr,th_cycles,downsample_factor,mseq_upsampled1,mseq_upsampled2,mseq_upsampled3,M_up, threshold_factor_seq1, threshold_factor_seq2, threshold_factor_seq3)
        
window_size = 30
corr_av_1=[]
corr_av_2=[]
corr_av_3=[]

t=[]
corr_final_first_seq=[0]
corr_final_second_seq=[0]
corr_final_third_seq=[0]
if __name__ == "__main__":
   visualizer = BinaryStateVisualizer()
   try:
    for i in range(num_reads):
        if(len(corr_av_1)>averaging_factor):
            corr_array_first_seq=corr_array_first_seq[-averaging_factor:]
            corr_array_second_seq=corr_array_second_seq[-averaging_factor:]
            corr_array_third_seq=corr_array_third_seq[-averaging_factor:]
        t=np.append(t,i)
        data = sdr.rx()
        Rx = data[0]
        Rx=Rx[::downsample_factor]
        envelope=np.abs(Rx)/2**12
        env_mean=np.mean(envelope)
        envelope-=env_mean
        envelope=envelope/np.max(envelope)

        corr_array_first_seq=np.append(corr_array_first_seq,np.max(np.abs(correlate(mseq_upsampled1, envelope, mode='full'))/M_up)) # saves all the peaks for seq 1 correlation
        corr_array_second_seq=np.append(corr_array_second_seq,np.max(np.abs(correlate(mseq_upsampled2, envelope, mode='full'))/M_up)) # saves all the peaks for seq 2 correlation
        corr_array_third_seq=np.append(corr_array_third_seq,np.max(np.abs(correlate(mseq_upsampled3, envelope, mode='full'))/M_up)) # saves all the peaks for seq 2 correlation

        corr_av_1=np.append(corr_av_1, np.mean(corr_array_first_seq))
        corr_av_2=np.append(corr_av_2, np.mean(corr_array_second_seq))
        corr_av_3=np.append(corr_av_3, np.mean(corr_array_third_seq))

        RIS_1_state= np.abs((corr_av_1[-1:]>th_1)*1)
        RIS_2_state= np.abs((corr_av_2[-1:]>th_2)*1)
        RIS_3_state= np.abs((corr_av_3[-1:]>th_3)*1)

        visualizer.update_states((int(RIS_1_state),int(RIS_2_state),int(RIS_3_state)))
            
        if i>window_size:
            t=t[-window_size:]
            corr_av_1=corr_av_1[-window_size:]
            corr_av_2=corr_av_2[-window_size:]
            corr_av_3=corr_av_3[-window_size:]


        line1.set_data(t,corr_av_1)
        line2.set_data(t,corr_av_2)
        line3.set_data(t,corr_av_3)
        line4.set_data(t,[th_1]*len(corr_av_1))
        line5.set_data(t,[th_2]*len(corr_av_2))
        line6.set_data(t,[th_3]*len(corr_av_3))

        bx.relim()
        bx.autoscale_view()

        plt.draw()
        plt.pause(0.01)
   except KeyboardInterrupt:
    pass
plt.ioff()
plt.show()

#plt.show()  # Show final figure (if needed)'''

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done
