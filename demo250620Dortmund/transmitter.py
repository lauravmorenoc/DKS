'''Transmitter'''

import adi
import numpy as np
import time

def find_sdr_uris():
    try:
        import subprocess
        import re

        result = subprocess.run(["iio_info", "-s"], capture_output=True, text=True)
        output = result.stdout

        uri_tx = None

        # Match lines like:
        # serial=104473... [usb:1.10.5]
        pattern = re.compile(r'serial=(\w+).*?\[(usb:[\d\.]+)\]')

        for match in pattern.finditer(output):
            serial = match.group(1)
            uri = match.group(2)

            print(f"Detected device → Serial: {serial}, URI: {uri}")

            if serial.endswith("51"):
                uri_tx = uri
                print("Assigned as Tx")

        if not uri_tx:
            raise RuntimeError(f"Could not find required SDR. Found TX={uri_tx}")

        return uri_tx
    
    except Exception as e:
        print(f"Error during SDR URI detection: {e}")
        raise

''' Variables '''

samp_rate = 5.3e5    # must be <=30.72 MHz if both channels are enabled (530000)
ts=1/samp_rate
NumSamples = 300000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
tx_gain = 0
fc0 = int(200e3)
cycles=1000000

'''Create Radios'''
uri_tx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)

sdr_tx.sample_rate = int(samp_rate)
sdr_tx.rx_rf_bandwidth = int(3 * fc0)
sdr_tx.rx_lo = int(rx_lo)
sdr_tx.gain_control_mode = "manual"
sdr_tx.rx_hardwaregain_chan0 = rx_gain
sdr_tx.rx_buffer_size = NumSamples
sdr_tx.tx_rf_bandwidth = int(3 * fc0)
sdr_tx.tx_lo = int(rx_lo)
sdr_tx.tx_cyclic_buffer = True
sdr_tx.tx_hardwaregain_chan0 = tx_gain
sdr_tx.tx_buffer_size = int(2**18)

# Transmit constant BPSK pattern (all 0s)
num_symbols = 10
sps = 16
x_symbols = np.ones(num_symbols)
samples = np.repeat(x_symbols, sps)
samples_tx = samples * (2**14)
sdr_tx.tx([samples_tx, samples_tx])

for i in range(cycles):
    print('Cycle: ', i)
    time.sleep(1)

sdr_tx.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done