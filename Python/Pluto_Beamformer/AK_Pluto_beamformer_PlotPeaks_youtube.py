"""
Jon Kraft, Oct 30 2022
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""

import adi
import matplotlib.pyplot as plt
import numpy as np
import time

'''Setup'''
samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 2.4e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 40
tx_lo = rx_lo
tx_gain = 0
fc0 = int(200e3)
phase_cal =-70
num_scans = 1000
Plot_Compass = True


''' Set distance between Rx antennas '''
d_wavelength = 0.5                  # distance between elements as a fraction of wavelength.  This is normally 0.5
wavelength = 3E8/rx_lo              # wavelength of the RF carrier
d = d_wavelength*wavelength         # distance between elements in meters
print("Set distance between Rx Antennas to ", int(d*1000), "mm")

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
#plt.plot(t,iq0)
sdr.tx([iq0,iq0])  # Send Tx data.

# Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
xf = np.fft.fftfreq(NumSamples, ts)
xf = np.fft.fftshift(xf)/1e6
signal_start = int(NumSamples*(samp_rate/2+fc0/2)/samp_rate)
signal_end = int(NumSamples*(samp_rate/2+fc0*2)/samp_rate)

def calcTheta(phase):
    # calculates the steering angle for a given phase delta (phase is in deg)
    # steering angle is theta = arcsin(c*deltaphase/(2*pi*f*d)
    arcsin_arg = np.deg2rad(phase)*3E8/(2*np.pi*rx_lo*d)
    arcsin_arg = max(min(1, arcsin_arg), -1)     # arcsin argument must be between 1 and -1, or numpy will throw a warning
    calc_theta = np.rad2deg(np.arcsin(arcsin_arg))
    return calc_theta

def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    #y = raw_data * np.transpose(win[np.newaxis])
    y = np.transpose(raw_data)*win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs
'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data = sdr.rx()
#################

plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#############
for i in range(num_scans):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_rlim(bottom=-20, top=0)
    ax.set_yticklabels([])
######## Rx #################
    data = sdr.rx()
    Rx_0=data[0]
    Rx_1=data[1]
    y_rx=np.asarray(data,dtype=np.complex_)
    #######################
    P_theta = []
    M=2
    theta0 = np.arange(-90, 90, 2)
    #for phase_delay in delay_phases:  
    for theta in theta0:  
        exponent = (np.arange(M)*np.pi*np.sin(np.deg2rad(theta+phase_cal)))
        a_theta =np.exp(-1j*exponent)
        y_wet = a_theta.conj().T@y_rx
        P_theta.append(10*np.log10(np.var(y_wet)))
    P_theta -=np.max(P_theta)
    steer_ang=theta0[np.argmax(P_theta)]
# #########  P l o t   #############
    ax.vlines(np.deg2rad(steer_ang),0,-20)
    ax.text(-2, -14, "{} deg".format(steer_ang))
    fig.canvas.flush_events()
    ax.clear()

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done


