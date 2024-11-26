

import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from graphics import *

'''Setup'''
samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 2.3e9 # Center frequency
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 40
tx_lo = rx_lo
tx_gain = -3 # -3dB attenuation
fc0 = int(200e3) # DC signal frequency
phase_cal = 0 # Phase calibration factor

'''Create Radio'''
sdr = adi.ad9361(uri='ip:192.168.2.1')

'''Configure properties for the Radio'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(fc0*3)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(NumSamples)
sdr._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
sdr.tx_rf_bandwidth = int(fc0*3)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain)
sdr.tx_hardwaregain_chan1 = int(-88)
sdr.tx_buffer_size = int(2**18)

'''Program Tx and Send Data'''
fs = int(sdr.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr.tx([iq0,iq0])  # Send Tx data.

# Assign frequency bins
xf = np.fft.fftfreq(NumSamples, ts)
xf = np.fft.fftshift(xf)/1e6

def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS=decibel full scale
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data = sdr.rx()

for i in range(1):
    data = sdr.rx() # grabs data
    Rx_0=data[0]   # separating the data
    Rx_1=data[1]
    
    peak_sum = []
    delay_phases = np.arange(-180, 180, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(phase_delay+phase_cal)) # makes the effect of adding phase on the response (first array)
        delayed_sum = dbfs(Rx_0 + delayed_Rx_1)

        # plot the FFT of the new delayed signal
        
        plt.plot(xf, delayed_sum)
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Rx0 + Rx1 [dBfs]")
        plt.ylim(top=0, bottom=-100)
        plt.text(-1, -10, "phase shift = {} deg".format(phase_delay))
        plt.draw()
         #plt.show()
        plt.pause(0.0001)
        plt.clf()
       
        ##

sdr.tx_destroy_buffer()




