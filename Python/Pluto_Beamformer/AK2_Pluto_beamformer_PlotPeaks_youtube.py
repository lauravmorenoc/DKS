"""
Jon Kraft, Oct 30 2022
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""

import adi
import matplotlib.pyplot as plt
import sys
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

########### SDR 1 ##################
# sdr = adi.ad9361(uri='ip:192.168.2.1')
# sdr2 = adi.Pluto(uri='ip:192.168.2.2')
sdr2=sdr=adi.ad9361(uri='usb:1.8.5')
#sdr2=adi.Pluto(uri='usb:1.9.5')

# sdr = adi.Pluto(uri='ip:192.168.2.2')

'''Configure properties for the Radio'''
sdr.rx_enabled_channels = [0, 1]
sdr.tx_enabled_channels = [0]
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

########## SDR 2 #################

'''Configure properties for the Radio'''
sdr2.rx_enabled_channels = [0]
sdr2.sample_rate = int(samp_rate)
sdr2.rx_rf_bandwidth = int(fc0*3)
sdr2.rx_lo = int(rx_lo)
sdr2.gain_control_mode = rx_mode
sdr2.rx_hardwaregain_chan0 = int(rx_gain0)
sdr2.rx_hardwaregain_chan1 = int(rx_gain1)
sdr2.rx_buffer_size = int(NumSamples)
sdr2._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
# sdr2.tx_rf_bandwidth = int(fc0*3)
# sdr2.tx_lo = int(tx_lo)
# sdr2.tx_cyclic_buffer = True
# sdr2.tx_hardwaregain_chan0 = int(tx_gain)
# sdr2.tx_hardwaregain_chan1 = int(-88)
# sdr2.tx_buffer_size = int(2**18)


'''Program Tx and Send Data'''
fs = int(sdr.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
#plt.plot(t,iq0)
# sdr.tx([iq0,iq0])  # Send Tx data.
sdr.tx(iq0) 

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
    T=1    
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    #data = sdr.rx()
    #data2 = sdr2.rx()

# fig, = plt.plot([])

# def update_line(fig, xv,yv):
#     fig.set_xdata(np.append(fig.get_xdata(), xv))
#     fig.set_ydata(NotImplementedError.append(fig.get_ydata(), yv))
#     plt.draw()

#plt.ion()

# x=np.array([0])
fig=plt.figure()
ax=fig.add_subplot(111)
x,y,x2,y2=[0],[0],[0],[0]
line1, = ax.plot([],[],label="SDR #1")
line2, = ax.plot([],[],label="SDR #2")
ax.legend()

ax.set_ylim(-100,0)
#################
for s in range(1000):
    #plt.clf()
#############
    #data = sdr.rx()
    data=data2 = sdr2.rx()

    Rx_0=data[0]
    Rx_1=data2
    #Rx_1=data[1]
    pow=10*np.log10(np.mean( (np.abs(Rx_0)/2**14)**2 ))
    pow2=10*np.log10(np.mean( (np.abs(Rx_1)/2**14)**2 ))
    # samp1 = np.mean( (np.abs(Rx_0[:len(Rx_0)//2])/2**14)**2 )
    # samp2 = np.mean( (np.abs(Rx_0[len(Rx_0)//2:])/2**14)**2 )
    # samp = np.append( samp1, samp2)
    # samp=10*np.log10(samp)
    #print(type(pow))
    #y[s]=pow
    if s>1000:
       x=x[1:len(x)]
       y=y[1:len(y)]
       x2=x2[1:len(x2)]
       y2=y2[1:len(y2)]

    y=np.append(y,pow)
    x=np.append(x,s+1)
    y2=np.append(y2,pow2)
    x2=np.append(x2,s+1)
    # update_line(fig,s,[pow])
    #print(y[s])
    #ax.set_ylim(-0.9*y[s],2*y[s])
    ax.set_xlim(0,s+50)
    # ax.set_xlim(0,s)
    line1.set_xdata(x)
    line1.set_ydata(y)

    line2.set_xdata(x2)
    line2.set_ydata(y2)

    #ax.set_label("SDR #1")

    #ax.autoscale(enable=True, axis="y", tight=False)
    # line1.set_xdata(x)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    #samp -=np.max(samp)
    # plt.plot(list(range(s,s+2)),samp)
    # plt.grid()
    # plt.show()
    # plt.savefig(sys.stdout.buffer)
    # sys.stdout.flush()
    plt.pause(0.1)
    t=1
    #######################
   

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done


