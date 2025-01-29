import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from graphics import *
import math


'''Setup'''
samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 2.3e9 # Center frequency
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 0
tx_lo = rx_lo
tx_gain = -10 # -10dB attenuation
fc0 = int(200e3) # DC signal frequency
phase_cal = 0 # Phase calibration factor

'''Create Radio'''
sdr = adi.ad9361(uri='ip:192.168.2.1')

'''Configure properties for the Radio'''
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(fc0*3)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
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

# Example read properties
print("RX LO %s" % (sdr.rx_lo))

# Generating Tx graph for comparison
fig=plt.figure()

cx=fig.add_subplot(311)
line1 = cx.plot(t[len(t)-50:len(t)],np.real(iq0)[len(t)-50:len(t)],label="Real Part")
line2 = cx.plot(t[len(t)-50:len(t)],np.imag(iq0)[len(t)-50:len(t)],label="Imaginary Part")
cx.grid(True)
cx.set_xlabel("Time")
cx.set_ylabel("Amplitude")
cx.set_title("Transmitted signal over time (Tx)")

bx=fig.add_subplot(312)
scatter_b = bx.scatter([],[],label="Tx Signal")
bx.grid(True)
bx.set_xlabel("Real part")
bx.set_ylabel("Imaginary part")
bx.set_xlim(-1, 1)
bx.set_ylim(-1, 1)
bx.set_title("Transmitted Signal (Tx)")
Tx=iq0
abs_val_tx=np.abs(Tx)
index_tx= np.where(abs_val_tx != 0)[0]
Tx_0 = Tx[index_tx] / abs_val_tx[index_tx]
pow_re_tx=np.real(Tx_0)
pow_im_tx=np.imag(Tx_0)
scatter_b = bx.scatter(pow_re_tx[len(pow_re_tx)-50:len(pow_re_tx)], pow_im_tx[len(pow_im_tx)-50:len(pow_im_tx)])
#'''

ax=fig.add_subplot(313)
x,y=[0],[0]
scatter = ax.scatter([],[],label="Rx Signal")
ax.grid(True)
ax.set_xlabel("Real part")
ax.set_ylabel("Imaginary part")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_title("Received Signal (Rx)")

for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data = sdr.rx()

for s in range(1000):

    data = sdr.rx()

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


sdr.tx_destroy_buffer()