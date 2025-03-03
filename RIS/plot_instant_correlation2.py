import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
#from graphics import *
import math

''' Functions '''

def conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,buffer_size):
    '''Configure properties for the Radio'''
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(fc0 * 3)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(NumSamples)
    sdr._rxadc.set_kernel_buffers_count(1)  # Set buffers to 1 to avoid stale data on Pluto
    fs=int(sdr.sample_rate)
    ts=1/fs
    return [fs, ts]

def get_fft(data):
    Rx_0=data
    NumSamples = len(Rx_0)
    win = np.hamming(NumSamples)
    y = Rx_0 * win
    sp = np.absolute(np.fft.fft(y))
    sp = sp[1:-1]
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / (np.sum(win)/2)    # Scale FFT by window and /2 since we are using half the FFT spectrum
    s_dbfs = 20*np.log10(s_mag/(2**12))     # Pluto is a 12 bit ADC, so use that to convert to dBFS
    xf = np.fft.fftfreq(NumSamples, ts)
    xf = np.fft.fftshift(xf[1:-1])/1e6
    return [xf, s_dbfs]

def show_fft(xf, s_dbfs):
    plt.clf()  # Clear the previous plot
    plt.plot(xf, s_dbfs, label=f"Scan {i+1}")  # Update plot
    plt.ylim([-100, 0])
    #plt.set_
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("dBfs")
    plt.legend()
    plt.xlim([-0.25, 0.25])
    plt.draw()
    plt.grid()
    plt.pause(0.01)  # Pause to allow real-time update


''' Variables '''

samp_rate = 1e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 10000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
tx_lo = rx_lo
tx_gain = -50 # -90 to 0 dB
fc0 = int(200e3)
RIS_seq_period=1e-4 


'''Create Radios'''

'''sdr=adi.ad9361(uri='ip:192.168.2.1')'''
sdr=adi.ad9361(uri='usb:1.8.5')
[fs, ts]=conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,NumSamples)


''' Pre-designed sequence '''

mseq1=np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
mseq2=np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])
#mseq1=np.array([1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,0,1])
#mseq2=np.array([0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,1,1,0])
M = len(mseq1)
amp=1 # signal amplitude
mseq1= np.where(mseq1 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
mseq2= np.where(mseq2 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
sps= int(RIS_seq_period/ts) # samples per symbol
mseq_upsampled1=np.repeat(mseq1, sps) # upsample sequence
mseq_upsampled2=np.repeat(mseq2, sps) # upsample sequence


'''Collect data'''

for r in range(20):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr.rx()

scanning_time=30 # seconds
pause=2
scanning_ts=pause+NumSamples*ts
peaks=[-60]
#plt.ion()  # Enable interactive mode
corr_final_first_seq=[0]
corr_final_second_seq=[0]
envelope_append=[0]

for i in range(2):
    del data
    corr_peaks_first_seq=[0]
    corr_peaks_second_seq=[0]
    data = sdr.rx()
    Rx = data[0]
    envelope=np.abs(Rx)/NumSamples
    env_mean=np.mean(envelope)
    envelope-=env_mean
    envelope_append=np.append(envelope_append, envelope)



plt.clf()  # Clear the previous plot
plt.plot(envelope_append/np.max(envelope_append), label="Envelope")
#plt.plot(mseq_upsampled1, label="mseq1")
#plt.ylim([-100, 0])
plt.xlabel("time (s)")
plt.ylabel("dBfs peak")
plt.legend()
#plt.xlim([-0.25, 0.25])
plt.draw()
plt.grid()
plt.show
    

    

#plt.ioff()  # Disable interactive mode when done
#plt.show()  # Show final figure (if needed)

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done