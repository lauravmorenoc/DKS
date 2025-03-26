'''Fine Frequency Synchronization'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

num_symbols = 100
sps = 8
bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's
pulse_train = np.array([])
for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    pulse_train = np.concatenate((pulse_train, pulse)) # add the 8 samples to the signal

# Create our raised-cosine filter
num_taps = 101
beta = 0.35
Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(-51, 52) # remember it's not inclusive of final number
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# Filter our signal, in order to apply the pulse shaping
samples_nd = np.convolve(pulse_train, h)

'''Adding a Delay'''
# Create and apply fractional delay filter
delay = 0.4 # fractional delay, in samples
N = 21 # number of taps
n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
h = np.sinc(n - delay) # calc filter taps
h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
samples = np.convolve(samples_nd, h) # apply filter

'''plt.figure(1)
plt.plot(samples_nd, '.-')
plt.plot(samples, '.-')
plt.grid(True)
plt.xlim([0, 150])
plt.title('Adding a time delay')
plt.show()'''

'''Adding a Frequency Offset'''

'''plt.subplot(2,1,1) # Before freq shift
plt.plot(np.real(samples), '.-', label='I')
plt.plot(np.imag(samples), '.-', label='Q')
plt.grid(True)
plt.legend()
plt.title('Before adding frequency offset')
plt.xlim([0, 150])'''

# apply a freq offset
fs = 1e6 # assume our sample rate is 1 MHz
fo = 1000 # simulate freq offset
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*len(samples), Ts) # create time vector
samples = samples * np.exp(1j*2*np.pi*fo*t) # perform freq shift
samples_no_corrected=samples

'''plt.subplot(2,1,2) # Before freq shift
plt.plot(np.real(samples), '.-', label='I')
plt.plot(np.imag(samples), '.-', label='Q')
plt.grid(True)
plt.legend()
plt.xlim([0, 150])
plt.title('After adding frequency offset')
plt.show()'''


'''Normal FFT: Before Squaring'''

psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))
'''plt.plot(f, psd)
plt.title('FFT before squaring (signal psd hides the offset)')
plt.show()'''

'''FFT After Squaring'''
N=2
samples_sqr = samples**N # we square it because it is a BPSK signal. It depends on the modulation order. For a QPSK it would be squaring two times (N=4)
psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))
'''plt.plot(f, psd)
plt.title('FFT after squaring (freq. offset peak visible)')
plt.show()'''

max_freq =f[np.argmax(psd)] 
print('Frequency offset: ', max_freq/N, ' Hz')

'''Applying Coarse Freq. Offset Correction'''
samples = samples * np.exp(-1j*2*np.pi*max_freq*t/N)
#samples = samples * np.exp(-1j*2*np.pi*fo*t)
samples_exp = samples**N # We square again to see if we removed the peak
psd = np.fft.fftshift(np.abs(np.fft.fft(samples_exp)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))
'''plt.plot(f, psd)
plt.title('FFT after squaring and removing freq. offset (coarse)')
plt.show()'''

'''Time Synchronization with Interpolation'''
samples_interpolated = signal.resample_poly(samples, 16, 1)

'''Mueller and Muller clock recovery technique'''
mu = 0 # initial estimate of phase of sample
k=0.3
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
samples = out # only include this line if you want to connect this code snippet with the Costas Loop later on

'''Fine Frequency Synchronization: Costas Loop technique, designed specifically for BPSK but can be upgraded for QPSK'''

N = len(samples)
phase = 0
freq = 0
# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=np.complex64)
freq_log = []
for i in range(N):
    out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq * fs / (2*np.pi*sps)) # convert from angular velocity to Hz for logging, adding sps divider because of time offset correction
    phase += freq + (alpha * error)

    # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    while phase >= 2*np.pi:
        phase -= 2*np.pi
    while phase < 0:
        phase += 2*np.pi

# Plot freq over time to see how long it takes to hit the right offset
plt.plot(freq_log,'.-')
plt.show()

'''Time Plots'''
plt.subplot(2,1,1) 
plt.plot(np.real(samples), '.-', label='I')
plt.plot(np.imag(samples), '.-', label='Q')
plt.grid(True)
plt.title('Before Costas Loop')
plt.subplot(2,1,2) 
plt.plot(np.real(out), '.-', label='I')
plt.plot(np.imag(out), '.-', label='Q')
plt.grid(True)
plt.title('After Costas Loop')
plt.show()

'''Scatter Plots'''
plt.subplot(1,2,1) 
plt.plot(np.real(samples_no_corrected[30:]), np.imag(samples_no_corrected[30:]), '.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(True)
plt.title('Before Sync')
plt.subplot(1,2,2) 
plt.plot(np.real(out[30:-6]), np.imag(out[30:-6]), '.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(True)
plt.title('After Coarse Freq + Time + Fine Freq Sync')
plt.show()