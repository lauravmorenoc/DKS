import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz
num_samps = 100000 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Create transmit waveform (simple sine wave, 16 samples per symbol)

num_symbols = 10000 # number of values
t = np.arange(num_symbols)/sample_rate
freq=1e4 # message frequency, Hz
x_symbols=np.cos(2*np.pi*freq*t)
#samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)
samples=x_symbols
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Plot freq domain
plt.figure(0)
plt.plot(t, x_symbols)
plt.xlabel("Time [s]")
plt.ylabel("Signal value")

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(samples) # start transmitting

# Receive samples
rx_samples = sdr.rx()
print(rx_samples)

# Stop transmitting
sdr.tx_destroy_buffer()

# Calculate power spectral density (frequency domain version of signal)
psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
psd_dB = 10*np.log10(psd)
f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))

# Plot time domain
plt.figure(1)
plt.plot(np.real(rx_samples))
plt.plot(np.imag(rx_samples))
#plt.plot(rx_samples)
plt.xlabel("Time")

# Plot freq domain
plt.figure(2)
plt.plot(f/1e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD")
plt.show()