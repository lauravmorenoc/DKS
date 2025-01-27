import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from graphics import *
import math

samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 2.4e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 40
tx_lo = rx_lo
tx_gain = 0
fc0 = int(200e3)
num_scans = 1000
Plot_Compass = False


'''Create Radios'''
sdr1=adi.ad9361(uri='usb:1.5.5')
sdr2=adi.ad9361(uri='usb:1.6.5')


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
fs = int(sdr1.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr1.tx([iq0,iq0])  # Send Tx data.


'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data = sdr2.rx()

for s in range(1000):

    data = sdr2.rx()

    Rx=data[0] # first channel
    abs_val=np.abs(Rx)
    index= np.where(abs_val != 0)[0]
    Rx_0 = Rx[index] / abs_val[index]

    pow_re=np.real(Rx_0)
    pow_im=np.imag(Rx_0)

    if s > 20:
        scatter.set_offsets(np.column_stack((pow_re[-50:], pow_im[-50:])))
    else:
        scatter = ax.scatter(pow_re[len(pow_re)-50:len(pow_re)], pow_im[len(pow_im)-50:len(pow_im)])

    plt.pause(0.1)
    t=1


sdr2.tx_destroy_buffer()