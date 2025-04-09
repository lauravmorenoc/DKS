import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal

'''Functions'''

def conf_sdr_rx(sdr, sample_rate, bandwidth, center_freq, mode, rx_gain,buffer_size):
    '''Configure properties for the Radio'''
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(bandwidth)
    sdr.rx_lo = int(center_freq)
    sdr.gain_control_mode = mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(buffer_size)
    sdr._rxadc.set_kernel_buffers_count(1)  # Set buffers to 1 to avoid stale data on Pluto
    fs=int(sdr.sample_rate)
    ts=1/fs
    return [fs, ts]

def conf_sdr_tx(sdr, sample_rate,center_freq, mode, tx_gain):
    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    sdr.gain_control_mode = mode
    sdr.tx_hardwaregain_chan0 = tx_gain # Increase to increase tx power, valid range is -90 to 0 dB
    sdr.tx_buffer_size = int(2**18)


def freq_offset(N, samples, fs):
    print('Finding coarse freq. offset')
    Ts=1/fs
    t = np.arange(0, Ts*len(samples), Ts)
    samples_sqr = samples**N
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))
    max_freq = f[np.argmax(psd)]
    print('Frequency offset: ', max_freq/N, ' Hz. Applying correction.')
    f_offset=max_freq/N
    return f_offset

'''Conf. Variables'''
sample_rate = 5.3e5 # Hz
center_freq = 5.3e9 # Hz
#center_freq = 915e6 # Hz
num_samps = 300000 # number of samples per call to rx()
gain_mode='manual'
rx_gain=0
tx_gain=-30 # Increase to increase tx power, valid range is -90 to 0 dB
#tx_gain=-50
N=2 # Order of the modulation

'''Conf. SDRs'''
sdr_tx = adi.ad9361(uri='usb:1.11.5')
sdr_rx = adi.ad9361(uri='usb:1.10.5')
conf_sdr_tx(sdr_tx,sample_rate,center_freq,gain_mode,tx_gain)
[fs, ts]=conf_sdr_rx(sdr_rx, sample_rate, sample_rate, center_freq, gain_mode, rx_gain,num_samps)

''' Create transmit waveform (BPSK, 16 samples per symbol) '''
num_symbols = 20
sps=16 # samples per symbol
plutoSDR_multiplier=2**14
#x_int = np.random.randint(0, 2, num_symbols) # 0 or 1
#x_int = np.array([0]*num_symbols)
x_int = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
x_degrees = x_int*360/2.0 # 0 or 180 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) # this produces our BPSK complex symbols
samples = np.repeat(x_symbols, sps) # 16 samples per symbol (rectangular pulses)
pulse_train=samples
samples_tx = samples*plutoSDR_multiplier # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Start the transmitter
sdr_tx.tx_cyclic_buffer = True # Enable cyclic buffers
sdr_tx.tx([samples_tx,samples_tx]) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr_rx.rx()


num_readings=5000

'''Plots'''
plt.ion()
fig=plt.figure()
bx=fig.add_subplot(111)
line1, = bx.plot([],[],marker='*')
bx.legend()
bx.set_title("Frequency offset over time")
bx.set_xlabel("Time Index")
bx.set_ylabel("Frequency offset (kHz)")
f_offset_array=[]

for i in range(num_readings):
    rx_2c = sdr_rx.rx()
    rx_samples=rx_2c[0]/plutoSDR_multiplier
    np.save("sdr_samples_100000_v2.npy", rx_samples)

    '''Coarse Frequency Synchronization'''
    f_offset=freq_offset(N, samples, fs)
    f_offset_array=np.append(f_offset_array, f_offset/1000)
    t=np.arange(0,len(f_offset_array),1)
    line1.set_data(t,f_offset_array)
    bx.relim()
    bx.autoscale_view()

    plt.draw()
    plt.pause(0.01)
    

# Stop transmitting
sdr_tx.tx_destroy_buffer()