import adi
import matplotlib.pyplot as plt
import numpy as np

# Create radio
sdr = adi.ad9361(uri='ip:192.168.2.1')
samp_rate = 30.72e6    # must be <=30.72 MHz if both channels are enabled
num_samps = 2**18      # number of samples per buffer.  Can be different for Rx and Tx
rx_lo = 1e9
rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
rx_gain0 = 10
rx_gain1 = 10


'''Configure Rx properties'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(num_samps)

# Example read properties
print("RX LO %s" % (sdr.rx_lo))

fig=plt.figure()
ax=fig.add_subplot(111)
x,y=[0],[0]
scatter = ax.scatter([],[],label="SDR #1")
ax.grid(True)
ax.set_xlabel("Real part")
ax.set_ylabel("Imaginary part")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data = sdr.rx()

for s in range(1000):

    data = sdr.rx()

    Rx=data[0] # first channel
    
    abs_val=np.abs(Rx)
    index= np.where(abs_val != 0)[0]
    #print(index)
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
if i>40: print('\a')    # for a long capture, beep when the script is done
