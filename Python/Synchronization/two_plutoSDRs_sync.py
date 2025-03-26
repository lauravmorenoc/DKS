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


def coarse_freq_corr(N, samples, fs):
    print('Finding coarse freq. offset')
    Ts=1/fs
    t = np.arange(0, Ts*len(samples), Ts)
    samples_sqr = samples**N 
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))
    max_freq = f[np.argmax(psd)] 
    print('Frequency offset: ', max_freq/N, ' Hz. Applying correction.')
    out = samples * np.exp(-1j*2*np.pi*max_freq*t/N)
    return out

def time_sync(samples,sps):
    samples_interpolated = signal.resample_poly(samples, 16, 1)
    mu = 0 # initial estimate of phase of sample
    k=0.3 # changes how fast the feedback loop reacts; a higher value will make it react faster, but with higher risk of stability issues
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + k*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    return out


'''Conf. Variables'''
sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz
num_samps = 100000 # number of samples per call to rx()
gain_mode='manual'
rx_gain=0
tx_gain=-30 # Increase to increase tx power, valid range is -90 to 0 dB
N=2 # Order of the modulation

'''Conf. SDRs'''
sdr_tx = adi.ad9361(uri='usb:1.7.5')
sdr_rx = adi.ad9361(uri='usb:1.8.5')
conf_sdr_tx(sdr_tx,sample_rate,center_freq,gain_mode,tx_gain)
[fs, ts]=conf_sdr_rx(sdr_rx, sample_rate, sample_rate, center_freq, gain_mode, rx_gain,num_samps)

''' Create transmit waveform (BPSK, 16 samples per symbol) '''
num_symbols = 1000
sps=16
plutoSDR_multiplier=2**14
x_int = np.random.randint(0, 2, num_symbols) # 0 or 1
x_degrees = x_int*360/2.0 # 0 or 180 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) # this produces our BPSK complex symbols
samples = np.repeat(x_symbols, sps) # 16 samples per symbol (rectangular pulses)
samples *= plutoSDR_multiplier # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Start the transmitter
sdr_tx.tx_cyclic_buffer = True # Enable cyclic buffers
sdr_tx.tx([samples,samples]) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr_rx.rx()

# Receive samples
rx_samples = sdr_rx.rx()
print(rx_samples)

# Stop transmitting
sdr_tx.tx_destroy_buffer()

plt.figure(1)
plt.plot(rx_samples, '.-')
plt.grid(True)
plt.xlim([0, 150])
plt.title('Received samples')
plt.show()

'''Coarse Frequency Synchronization'''
samples=coarse_freq_corr(N, rx_samples, fs)

'''Scatter Plots'''
plt.subplot(1,2,1) 
plt.plot(np.real(rx_samples[30:]), np.imag(rx_samples[30:]), '.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(True)
plt.title('Before Coarse Freq Sync')
plt.subplot(1,2,2)
plt.plot(np.real(samples[30:-6]), np.imag(samples[30:-6]), '.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(True)
plt.title('After Coarse Freq Sync')
plt.show()

'''Time Sync'''
#samples=time_sync(samples,sps)