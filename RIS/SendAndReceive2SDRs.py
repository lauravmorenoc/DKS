import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
#from graphics import *
import math

samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 0 # 0 to 50 dB
rx_gain1 = 0
tx_lo = rx_lo
tx_gain = -50 # -90 to 0 dB
fc0 = int(200e3)
num_scans = 1000
Plot_Compass = False


'''Create Radios'''
sdr1=adi.ad9361(uri='usb:1.18.5')
sdr2=adi.ad9361(uri='usb:1.14.5') # Rx
#sdr1=sdr2=adi.ad9361(uri='ip:192.168.2.1')

'''Configure properties for the Radio'''
sdr1.sample_rate = sdr2.sample_rate = int(samp_rate)
sdr1.rx_rf_bandwidth = sdr2.rx_rf_bandwidth = int(fc0 * 3)
sdr1.rx_lo = sdr2.rx_lo = int(rx_lo)
sdr1.gain_control_mode = sdr2.gain_control_mode = rx_mode
sdr1.rx_hardwaregain_chan0 = sdr2.rx_hardwaregain_chan0 = int(rx_gain0)
sdr1.rx_buffer_size = sdr2.rx_buffer_size = int(NumSamples)
sdr1._rxadc.set_kernel_buffers_count(1)  # Set buffers to 1 to avoid stale data on Pluto
sdr2._rxadc.set_kernel_buffers_count(1)
sdr1.tx_rf_bandwidth = int(fc0*3)
sdr1.tx_lo = int(tx_lo)
sdr1.tx_cyclic_buffer = True
sdr1.tx_hardwaregain_chan0 = int(tx_gain)
sdr1.tx_buffer_size = int(2**18)

'''Program Tx and Send Data'''
a = 1 #symbo=a+jb
b= 1
fs = int(sdr1.sample_rate)
Tx_time=0.25e-3
#N=Tx_time*fs
N = 2**16
ts = 1 / float(fs)
#t = np.arange(0, N * ts, ts)
t = np.arange(0, Tx_time, ts)

i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = (a)*i0 + (b)*1j * q0
#sdr1.tx([iq0,iq0])  # Send Tx data.




# Collect data
for r in range(20):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr2.rx()

num_scans=10000
plt.ion()  # Enable interactive mode

for i in range(num_scans):
    del data
    data = sdr2.rx()
    Rx_0 = data[0]
    NumSamples = len(Rx_0)
    '''
    #psd = np.abs(np.fft.fftshift(np.fft.fft(Rx_0)))**2
    #psd_dB = 10*np.log10(psd)
    #f = np.linspace(samp_rate/-2, samp_rate/2, len(psd))
    '''


    win = np.hamming(NumSamples)
    y = Rx_0 * win
    sp = np.absolute(np.fft.fft(y))
    sp = sp[1:-1]
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / (np.sum(win)/2)    # Scale FFT by window and /2 since we are using half the FFT spectrum
    s_dbfs = 20*np.log10(s_mag/(2**12))     # Pluto is a 12 bit ADC, so use that to convert to dBFS
    xf = np.fft.fftfreq(NumSamples, ts)
    xf = np.fft.fftshift(xf[1:-1])/1e6

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
    

plt.ioff()  # Disable interactive mode when done
plt.show()  # Show final figure (if needed)

sdr2.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done
