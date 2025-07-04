"""
Jon Kraft, Oct 30 2022
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""

import adi
import matplotlib.pyplot as plt
import numpy as np


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
phase_cal =-174
num_scans = 1000
Plot_Compass = False
NRxAntennas= 4 # Number of receiving antennas

''' Set distance between Rx antennas '''
d_wavelength = 0.5                  # distance between elements as a fraction of wavelength.  This is normally 0.5
wavelength = 3E8/rx_lo              # wavelength of the RF carrier
d = d_wavelength*wavelength         # distance between elements in meters
print("Set distance between Rx Antennas to ", int(d*1000), "mm")

'''Create Radios'''
#sdr1 = adi.ad9361(uri='ip:192.168.2.1')
#sdr2 = adi.ad9361(uri='ip:169.254.241.224') # Add IP Adress
sdr1=adi.ad9361(uri='usb:1.5.5')
sdr2=adi.ad9361(uri='usb:1.6.5')


'''Configure properties for the Radio'''
sdr1.rx_enabled_channels = [0, 1]
sdr2.rx_enabled_channels = [0, 1]
sdr1.sample_rate = sdr2.sample_rate = int(samp_rate)
sdr1.rx_rf_bandwidth = sdr2.rx_rf_bandwidth = int(fc0 * 3)
sdr1.rx_lo = sdr2.rx_lo = int(rx_lo)
sdr1.gain_control_mode = sdr2.gain_control_mode = rx_mode
sdr1.rx_hardwaregain_chan0 = sdr2.rx_hardwaregain_chan0 = int(rx_gain0)
sdr1.rx_hardwaregain_chan1 = sdr2.rx_hardwaregain_chan1 = int(rx_gain1)
sdr1.rx_buffer_size = sdr2.rx_buffer_size = int(NumSamples)
sdr1._rxadc.set_kernel_buffers_count(1)  # Set buffers to 1 to avoid stale data on Pluto
sdr2._rxadc.set_kernel_buffers_count(1)
sdr1.tx_rf_bandwidth = int(fc0*3)
sdr1.tx_lo = int(tx_lo)
sdr1.tx_cyclic_buffer = True
sdr1.tx_hardwaregain_chan0 = int(tx_gain)
sdr1.tx_hardwaregain_chan1 = int(-88)
sdr1.tx_buffer_size = int(2**18)


'''Program Tx and Send Data'''
fs = int(sdr1.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr1.tx([iq0,iq0])  # Send Tx data.

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
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

 # a_thet=[1 exp(-j*pi*sin(theta)) exp(-j*2*pi*sin(theta))]

'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data1 = sdr1.rx()
    data2 = sdr2.rx()

for i in range(num_scans):
    data = sdr1.rx()
    Rx=np.array([data1[0], data1[1], data2[0], data2[1]]) # saves data from the for antennas

    peak_sum = []
    delay_phases = np.arange(-180, 180, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delay=np.array([1, np.exp(1j*np.deg2rad(phase_delay+phase_cal))])
        if NRxAntennas>2:
            for m in range(2, NRxAntennas):
                delay= np.concatenate((delay, m*np.exp(1j*np.deg2rad(phase_delay+phase_cal))))

        delayed_Rx = Rx * delay
        delayed_sum = dbfs(sum(delayed_Rx))
        peak_sum.append(np.max(delayed_sum[signal_start:signal_end]))
    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    steer_angle = int(calcTheta(peak_delay))
    if Plot_Compass==False:
        plt.plot(delay_phases, peak_sum)
        plt.axvline(x=peak_delay, color='r', linestyle=':')
        plt.text(-180, -26, "Peak signal occurs with phase shift = {} deg".format(round(peak_delay,1)))
        plt.text(-180, -28, "If d={}mm, then steering angle = {} deg".format(int(d*1000), steer_angle))
        plt.ylim(top=0, bottom=-30)        
        plt.xlabel("phase shift [deg]")
        plt.ylabel("Rx0 + Rx1 [dBfs]")
        plt.draw()
        #plt.show()
        plt.pause(0.01)
        plt.clf()

    else:
        fig = plt.figure(figsize=(3,3))
        ax = plt.subplot(111,polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_rlim(bottom=-20, top=0)
        ax.set_yticklabels([])
        ax.vlines(np.deg2rad(steer_angle),0,-20)
        ax.text(-2, -14, "{} deg".format(steer_angle))
        plt.draw()
        #plt.show()
        plt.pause(0.01)
        plt.clf()

sdr1.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done

