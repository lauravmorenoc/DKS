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
ax.set_xlabel("Real part")
ax.set_ylabel("Imaginary part")

for s in range(1000):

    data = sdr.rx()

    Rx_0=data
    Rx_0=Rx_0/np.abs(Rx_0)

    pow_re=np.mean(np.real(Rx_0))
    pow_im=np.mean(np.imag(Rx_0))

    if s>100:
       x=x[1:len(x)]
       y=y[1:len(y)]

    y=np.append(y,pow_im)
    x=np.append(x,pow_re)

    scatter = ax.scatter(x, y)

    plt.pause(0.1)
    t=1

   
sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done
