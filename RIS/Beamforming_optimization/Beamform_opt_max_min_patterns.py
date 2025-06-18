import serial
import time
import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



# ===================== USER PARAMETERS =====================
ris_port = 'COM18'
baudrate = 115200
reads_per_check = 3
group_size = 16
change_period= 1 # seconds
num_rows = 16
num_cols = 16
rx_lo = 5.3e9
sample_rate = 5.3e5
NumSamples = 300000
rx_gain = 0
tx_gain = 0
# ===========================================================


def ris_init(port, baudrate):
    ris = serial.Serial(port, baudrate, timeout=1)
    ris.write(b'!Reset\n')
    time.sleep(1)
    while ris.in_waiting > 0:
        response = ris.readline().decode().strip()
        print(f"Reset Response: {response}")
        time.sleep(0.1)
    return ris


def generate_pattern(state_array):
    bitstring = ''.join(['1' if x else '0' for x in state_array])
    hexstring = hex(int(bitstring, 2))[2:].zfill(64)
    return '!0x' + hexstring


def send_pattern(ris, pattern):
    ris.write((pattern + '\n').encode())
    time.sleep(0.05)
    if ris.in_waiting > 0:
        return ris.readline().decode().strip()
    return None


def measure_power(sdr_rx, NumSamples, reads_per_check):
    powers = []
    for _ in range(reads_per_check):
        data = sdr_rx.rx()
        Rx_0 = data[0]
        y = Rx_0 * np.hamming(NumSamples)
        sp = np.abs(np.fft.fftshift(np.fft.fft(y)[1:-1]))
        mag = np.abs(sp) / (np.sum(np.hamming(NumSamples)) / 2)
        dbfs = 20 * np.log10(mag / (2**12))
        peak_power = np.max(dbfs)
        powers.append(peak_power)
    return np.mean(powers)


def find_sdr_uris():
    try:
        import subprocess
        import re

        result = subprocess.run(["iio_info", "-s"], capture_output=True, text=True)
        output = result.stdout

        uri_tx = None
        uri_rx = None

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
            elif serial.endswith("74"):
                uri_rx = uri
                print("Assigned as Rx")

        if not uri_tx or not uri_rx:
            raise RuntimeError(f"Could not find both required SDRs. Found TX={uri_tx}, RX={uri_rx}")

        return uri_tx, uri_rx

    except Exception as e:
        print(f"Error during SDR URI detection: {e}")
        raise


# ---- MAIN SCRIPT ----
ris = ris_init(ris_port, baudrate)
state = [0] * (num_rows * num_cols)  # initial state: all OFF
send_pattern(ris, generate_pattern(state))

# Configure SDR
uri_tx, uri_rx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)
sdr_rx = adi.ad9361(uri=uri_rx)
sdr_tx.sample_rate = sdr_rx.sample_rate = int(sample_rate)
sdr_tx.rx_rf_bandwidth = sdr_rx.rx_rf_bandwidth = int(3 * 0)
sdr_tx.rx_lo = sdr_rx.rx_lo = int(rx_lo)
sdr_tx.gain_control_mode = sdr_rx.gain_control_mode = "manual"
sdr_tx.rx_hardwaregain_chan0 = sdr_rx.rx_hardwaregain_chan0 = rx_gain
sdr_tx.rx_buffer_size = sdr_rx.rx_buffer_size = NumSamples
sdr_tx.tx_rf_bandwidth = int(3 * 0)
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

# Compute group top-left positions
group_side = int(np.sqrt(group_size))
power_history = []

# Initial power
current_power = measure_power(sdr_rx, NumSamples, reads_per_check)
power_history.append(current_power)


# ============================= MAXIMIZING =============================

for row in range(0, num_rows, group_side):
    for col in range(0, num_cols, group_side):
        print(f"\nChecking group at ({row}, {col})")

        # Get indices of this 4x4 group
        indices = []
        for r in range(group_side):
            for c in range(group_side):
                i = (row + r) * num_cols + (col + c)
                indices.append(i)

        # Flip group bits
        for idx in indices:
            state[idx] ^= 1  # toggle

        # Send new pattern and measure power
        send_pattern(ris, generate_pattern(state))
        new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

        print(f"Old power: {current_power:.2f} dBFS, New power: {new_power:.2f} dBFS")

        if new_power >= current_power:
            print("→ Change accepted.")
            current_power = new_power
        else:
            print("→ Change reverted.")
            # revert changes
            for idx in indices:
                state[idx] ^= 1
            send_pattern(ris, generate_pattern(state))

        power_history.append(current_power)
        time.sleep(change_period)


# Save or plot power history
plt.figure()
plt.plot(power_history, marker='o')
plt.xlabel("Group index")
plt.ylabel("Max Power (dBFS)")
plt.title("Power evolution during RIS optimization")
plt.grid()
plt.show(block=False)

# Saving in text file
x = 1
y = 1

# Generate the pattern string
hex_pattern = generate_pattern(state)

# Format the full line to write
#line_to_write = f"{hex_pattern} [{x},{y}]\n"
line_to_write = f"{hex_pattern}\n"

# Append to file instead of overwriting
with open("optimized_max_ris_pattern_hex.txt", "a") as f:  # 'a' = append mode
    f.write(line_to_write)


# ============================= MINIMIZING =============================

print('Minimizing now')
power_history = []

# Initial power
current_power = measure_power(sdr_rx, NumSamples, reads_per_check)
power_history.append(current_power)

# Get min power pattern
for row in range(0, num_rows, group_side):
    for col in range(0, num_cols, group_side):
        print(f"\nChecking group at ({row}, {col})")

        # Get indices of this 4x4 group
        indices = []
        for r in range(group_side):
            for c in range(group_side):
                i = (row + r) * num_cols + (col + c)
                indices.append(i)

        # Flip group bits
        for idx in indices:
            state[idx] ^= 1  # toggle

        # Send new pattern and measure power
        send_pattern(ris, generate_pattern(state))
        new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

        print(f"Old power: {current_power:.2f} dBFS, New power: {new_power:.2f} dBFS")

        if new_power <= current_power:
            print("→ Change accepted.")
            current_power = new_power
        else:
            print("→ Change reverted.")
            # revert changes
            for idx in indices:
                state[idx] ^= 1
            send_pattern(ris, generate_pattern(state))

        power_history.append(current_power)
        time.sleep(change_period)


# Save or plot power history
plt.figure()
plt.plot(power_history, marker='o')
plt.xlabel("Group index")
plt.ylabel("Max Power (dBFS)")
plt.title("Power evolution during RIS optimization")
plt.grid()
plt.show()

# Saving in text file
hex_pattern = generate_pattern(state)
line_to_write = f"{hex_pattern} [{x},{y}]\n"
with open("optimized_min_ris_pattern_hex.txt", "a") as f:  # 'a' = append mode
    f.write(line_to_write)

ris.close()