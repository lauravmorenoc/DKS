import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal
import time
from collections import deque

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
    #off_corr=np.exp(-1j*2*np.pi*max_freq*t/N)
    #out = samples * off_corr[-1:]
    out = samples * np.exp(-1j*2*np.pi*max_freq*t/N)

    samples_sqr = out**N
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))
    max_freq = f[np.argmax(psd)]
    f_residual=max_freq/N
    #print('Frequency residual: ',f_residual)
    return out, f_residual

def time_sync(samples,sps):
    samples_interpolated = signal.resample_poly(samples, 16, 1)
    mu = 0 # initial estimate of phase of sample
    k=0.3 # changes how fast the feedback loop reacts; a higher value will make it react faster, but with higher risk of stability issues 0.3,0.5,2
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
sample_rate = 3e6 # Hz
center_freq = 5.3e9 # Hz
#center_freq = 915e6 # Hz
num_samps = 50000 # number of samples per call to rx()
gain_mode='manual'
rx_gain=0
tx_gain=0 # Increase to increase tx power, valid range is -90 to 0 dB (it was -30 before)
#tx_gain=-50
N=2 # Order of the modulation

'''Conf. SDRs'''
sdr_tx = adi.ad9361(uri='usb:1.7.5')
sdr_rx = adi.ad9361(uri='usb:1.8.5')
conf_sdr_tx(sdr_tx,sample_rate,center_freq,gain_mode,tx_gain)
[fs, ts]=conf_sdr_rx(sdr_rx, sample_rate, sample_rate, center_freq, gain_mode, rx_gain,num_samps)

''' Create transmit waveform (BPSK, 16 samples per symbol) '''
num_symbols = 20
sps=16 # samples per symbol
plutoSDR_multiplier=2**14
#x_int = np.random.randint(0, 2, num_symbols) # 0 or 1
#x_int = np.array([0]*num_symbols)
#x_int = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
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


# Parameters
num_readings = 1000
max_plot_length = 200
downsample_len = 100

# Buffers to store the signal over time (with rolling window)
mag_log = deque(maxlen=max_plot_length)   # Magnitude
real_log = deque(maxlen=max_plot_length)  # Real part
imag_log = deque(maxlen=max_plot_length)  # Imaginary part
phase_log = deque(maxlen=max_plot_length) # Phase (angle)

# Set up interactive plotting
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Two subplots: [Amplitude, Phase]

# --- Left Plot: Magnitude, Real and Imaginary components ---
line_mag, = axs[0].plot([], [], 'b-', label='Magnitude')
line_real, = axs[0].plot([], [], 'r--', label='Real part')
line_imag, = axs[0].plot([], [], 'g--', label='Imag part')
axs[0].set_title("Signal Amplitude Components")
axs[0].set_ylim([-0.1, 0.1])  # Adjust based on signal amplitude range
axs[0].set_xlim([0, max_plot_length])
axs[0].set_xlabel("Sample index")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)
axs[0].legend()

# --- Right Plot: Phase ---
line_phase, = axs[1].plot([], [], 'm-', label='Phase')
axs[1].set_title("Phase (degrees)")
axs[1].set_ylim([-180, 180])
axs[1].set_xlim([0, max_plot_length])
axs[1].set_xlabel("Sample index")
axs[1].set_ylabel("Phase (°)")
axs[1].grid(True)

# --- Main loop ---
for i in range(num_readings):
    rx_2c = sdr_rx.rx()                      # Receive complex samples from SDR
    rx_samples = rx_2c[0] / plutoSDR_multiplier  # Normalize the samples

    # Downsample to 100 samples
    step = len(rx_samples) // downsample_len
    rx_samples_ds = rx_samples[::step][:downsample_len]

    # Extract magnitude, real, imaginary, and phase components
    mag = np.abs(rx_samples_ds)
    real = np.real(rx_samples_ds)
    imag = np.imag(rx_samples_ds)
    phase = np.degrees(np.angle(rx_samples_ds))  # Convert phase to degrees

    # Append new data to rolling buffers
    mag_log.extend(mag)
    real_log.extend(real)
    imag_log.extend(imag)
    phase_log.extend(phase)

    # Create x-axis values based on current length
    x_vals = np.arange(len(mag_log))

    # Update each line plot
    line_mag.set_data(x_vals, list(mag_log))
    line_real.set_data(x_vals, list(real_log))
    line_imag.set_data(x_vals, list(imag_log))
    line_phase.set_data(x_vals, list(phase_log))

    # Update x-axis limits dynamically
    axs[0].set_xlim([0, len(mag_log)])
    axs[1].set_xlim([0, len(phase_log)])

    # Refresh plots
    plt.draw()
    plt.pause(0.01)

# Stop transmitting
sdr_tx.tx_destroy_buffer()