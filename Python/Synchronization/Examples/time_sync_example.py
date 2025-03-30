'''Time Sync Example'''

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

plt.figure(1)
plt.plot(samples_nd, '.-', label='Original')
plt.plot(samples, '.-', label='Delayed')
plt.grid(True)
plt.xlim([0, 150])
plt.legend()
plt.show()

'''Adding a Frequency Offset'''

fs = 1e6 # assume our sample rate is 1 MHz
fo = 1000 # simulate freq offset
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*len(samples), Ts) # create time vector
samples = samples * np.exp(1j*2*np.pi*fo*t) # perform freq shift

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
#samples = out # only include this line if you want to connect this code snippet with the Costas Loop later on

'''Time Plots'''
plt.subplot(3,1,1) 
plt.plot(pulse_train, '.-')
plt.grid(True)
plt.title('Original BPSK symbols')
plt.subplot(3,1,2) 
plt.plot(np.real(samples_nd), '.-')
plt.grid(True)
plt.title('Samples after pulse shaping but before the synchronizer')
plt.subplot(3,1,3) 
plt.plot(np.real(out), '.-', label='I')
plt.plot(np.imag(out), '.-', label='Q')
plt.legend()
plt.grid(True)
plt.title('Output of the symbol synchronizer (1sps)')
plt.show()

'''Scatter Plots'''
plt.subplot(1,2,1) 
plt.plot(np.real(samples[30:]), np.imag(samples[30:]), '.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(True)
plt.title('Before Time Sync')
plt.subplot(1,2,2) 
plt.plot(np.real(out[30:-6]), np.imag(out[30:-6]), '.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid(True)
plt.title('After Time Sync')
plt.show()