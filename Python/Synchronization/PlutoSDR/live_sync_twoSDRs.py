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
    #off_corr=np.exp(-1j*2*np.pi*max_freq*t/N)
    #out = samples * off_corr[-1:]
    out = samples * np.exp(-1j*2*np.pi*max_freq*t/N)

    samples_sqr = out**N
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))
    max_freq = f[np.argmax(psd)]
    f_residual=max_freq/N
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

def fine_freq_corr(samples, sps, f_residual):
    N = len(samples)
    phase = 0
    #freq = 0
    freq = 2 * np.pi * f_residual / fs
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 10 # 0.132, modified to 0.09 for 916 MHz, cable: 1, wireless: 10
    beta = alpha*0.007 # 0.00932, modified to 0.0095 for 916 MHz, cable: 0.03, wireless:0.007
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    error_log = []
    phase_log=[]
    for i in range(N):
        out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
        error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
        error_log.append(error)
        phase_log.append(phase)

        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        freq_log.append(freq * fs / (2*np.pi*sps)) # convert from angular velocity to Hz for logging, adding sps divider because of time offset correction
        phase += freq + (alpha * error)

        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

    return out,phase_log

'''Conf. Variables'''
sample_rate = 3e6 # Hz
center_freq = 5.3e9 # Hz
#center_freq = 915e6 # Hz
num_samps = 50000 # number of samples per call to rx()
gain_mode='manual'
rx_gain=0
tx_gain=0 # Increase to increase tx power, valid range is -90 to 0 dB
#tx_gain=-50
N=2 # Order of the modulation

'''Conf. SDRs'''
sdr_tx = adi.ad9361(uri='usb:1.29.5')
sdr_rx = adi.ad9361(uri='usb:1.28.5')
conf_sdr_tx(sdr_tx,sample_rate,center_freq,gain_mode,tx_gain)
[fs, ts]=conf_sdr_rx(sdr_rx, sample_rate, sample_rate, center_freq, gain_mode, rx_gain,num_samps)

''' Create transmit waveform (BPSK, 16 samples per symbol) '''
num_symbols = 20
sps=16 # samples per symbol
plutoSDR_multiplier=2**14
#x_int = np.random.randint(0, 2, num_symbols) # 0 or 1
#x_int = np.array([0]*num_symbols)
x_int = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
x_degrees = x_int*360/2.0 # 0 or 180 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) # this produces our BPSK complex symbols
samples = np.repeat(x_symbols, sps) # 16 samples per symbol (rectangular pulses)
pulse_train=samples[::16]

'''plt.figure(1)
plt.plot(pulse_train[-6000:], '.-')
plt.title('Sent samples')
plt.show()'''
samples_tx = samples*plutoSDR_multiplier # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Start the transmitter
sdr_tx.tx_cyclic_buffer = True # Enable cyclic buffers
sdr_tx.tx([samples_tx,samples_tx]) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr_rx.rx()


num_readings=50

'''Plots'''
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sc1 = axs[0].scatter([], [], s=3)
sc2 = axs[1].scatter([], [], s=3)

fig_pulses, ax_pulses = plt.subplots(figsize=(10, 4))
#line_mag, = ax_pulses.plot([], [], 'b-', label='Original pulses')
line_real, = ax_pulses.plot([], [], 'r', marker='o', label='Real part')
line_imag, = ax_pulses.plot([], [], 'g', marker='o', label='Imag part')
ax_pulses.set_title("Time sync plots")
ax_pulses.set_xlabel("Sample index")
ax_pulses.set_ylabel("Amplitude")
ax_pulses.grid(True)
ax_pulses.legend()


axs[0].set_title("Before Sync")
axs[1].set_title("After Sync")
for ax in axs:
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.grid(True)

for i in range(num_readings):
    rx_2c = sdr_rx.rx()
    rx_samples=rx_2c[0]/plutoSDR_multiplier
    np.save("sdr_samples_100000_v2.npy", rx_samples)

    '''Coarse Frequency Synchronization'''
    rx_samples_corrected, f_residual=coarse_freq_corr(N, rx_samples, fs)

    '''Time Sync'''
    rx_samples_corrected=time_sync(rx_samples_corrected,sps)

    '''Fine Frequency Synchronization'''
    print('f_residual= ', f_residual)
    rx_samples_corrected, phase_log=fine_freq_corr(rx_samples_corrected, sps, f_residual)
    #rx_samples_corrected=samples

    '''Scatter Plots'''
    rx_samples=rx_samples/max(abs(rx_samples))
    rx_samples_corrected=rx_samples_corrected/max(abs(rx_samples_corrected))
    x1 = np.column_stack((np.real(rx_samples[::16]), np.imag(rx_samples[::16])))
    x2 = np.column_stack((np.real(rx_samples_corrected[-1562:]), np.imag(rx_samples_corrected[-1562:]))) # 4500
    #print(len(rx_samples_corrected))
    sc1.set_offsets(x1)
    sc2.set_offsets(x2)

    '''Time Plots
    t=np.arange(0,len(rx_samples_corrected[-100:]),1)
    #line_mag.set_data(t, pulse_train[-1562:])
    line_real.set_data(t, np.real(rx_samples_corrected[-100:]))
    line_imag.set_data(t, np.imag(rx_samples_corrected[-100:]))
    ax_pulses.relim()
    ax_pulses.autoscale_view()'''
    plt.draw()
    plt.pause(0.25)

    #plt.pause(0.01)

# Stop transmitting
sdr_tx.tx_destroy_buffer()