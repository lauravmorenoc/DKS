import serial
import time
import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import re


# ===================== USER PARAMETERS =====================
ris_ports = ['COM18', 'COM19']     # <-- both ports here
baudrate  = 115200
reads_per_check = 3
#group_size = 16
change_period= 1 # seconds
num_rows = 16
num_cols = 16
rx_lo = 5.3e9
sample_rate = 5.3e5
NumSamples = 300000
rx_gain = 30
tx_gain = 0
group_len   = 8      # width of a stripe
group_height = 1     # stripes are one-row tall

pos = 2 # 60°
# ===========================================================

FILE_PATH_MAX = "optimized_max_ris_pattern_hex.txt"
FILE_PATH_MIN = "optimized_min_ris_pattern_hex.txt"
LINE_RE   = re.compile(r'^(.*)\s+\[(\d+)]\s*$')

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


def send_all_patterns(ris_list, states):
    """Generate and send the pattern for every board."""
    for ris, st in zip(ris_list, states):
        send_pattern(ris, generate_pattern(st))


def save_ris_patterns(pattern_pair, pos, path):
    """pattern_pair = (pat_A, pat_B)  →  '<patA> <patB> [pos]' """
    records = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = LINE_RE.match(line)
                if m:
                    records[int(m.group(2))] = m.group(1)

    records[pos] = f"{pattern_pair[0]} {pattern_pair[1]}"
    with open(path, "w", encoding="utf-8") as f:
        for p in sorted(records):
            f.write(f"{records[p]} [{p}]\n")


def measure_power(sdr_rx, NumSamples, reads_per_check):
    powers = []
    for _ in range(reads_per_check):
        data = sdr_rx.rx()
        Rx_0 = data[0]
        '''y = Rx_0 * np.hamming(NumSamples)
        sp = np.abs(np.fft.fftshift(np.fft.fft(y)[1:-1]))
        mag = np.abs(sp) / (np.sum(np.hamming(NumSamples)) / 2)
        dbfs = 20 * np.log10(mag / (2**12))
        peak_power = np.max(dbfs)'''
        peak_power = np.mean(np.abs( Rx_0 )**2)
        peak_power = 20 * np.log10(peak_power / (2**12))
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


def optimize_dual_ris(states, mode, ris_list, sdr_rx, *,       # named *
                      generate_pattern, send_pattern, measure_power,
                      group_len, num_rows, num_cols, change_period):

    power_hist = []
    cur_power  = measure_power(sdr_rx, NumSamples, reads_per_check)
    power_hist.append(cur_power)

    for row in range(num_rows):
        for col in range(0, num_cols, group_len):
            for board in (0, 1):                    # panel A then B
                idxs = [row*num_cols + (col+off) for off in range(group_len)]

                # toggle stripe on the chosen board
                for i in idxs:
                    states[board][i] ^= 1

                send_all_patterns(ris_list, states)
                new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

                keep = ((mode == "max" and new_power >= cur_power) or
                        (mode == "min" and new_power <= cur_power))
                if keep:
                    cur_power = new_power
                else:                               # revert that stripe
                    for i in idxs:
                        states[board][i] ^= 1
                    send_all_patterns(ris_list, states)

                power_hist.append(cur_power)
                time.sleep(change_period)

    pat_A = generate_pattern(states[0])
    pat_B = generate_pattern(states[1])
    return states, (pat_A, pat_B), power_hist



# ---- MAIN SCRIPT ----
ris_list = [ris_init(p, baudrate) for p in ris_ports]

states = [[0]*(num_rows*num_cols) for _ in ris_list]  # [state_A, state_B]
send_all_patterns(ris_list, states)

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


# ============================= MAXIMIZING =============================

print('Maximizing')
states, pats, p_hist = optimize_dual_ris(states, "max", ris_list, sdr_rx,
                                         generate_pattern=generate_pattern,
                                         send_pattern=send_pattern,
                                         measure_power=measure_power,
                                         group_len=group_len,
                                         num_rows=num_rows,
                                         num_cols=num_cols,
                                         change_period=change_period)

save_ris_patterns(pats, pos, path=FILE_PATH_MAX)


# Save or plot power history
plt.figure()
plt.plot(p_hist, marker='o')
plt.xlabel("Group index")
plt.ylabel("Max Power (dBFS)")
plt.title("Power evolution during RIS optimization")
plt.grid()
plt.show(block=False)


# ============================= MINIMIZING =============================

print('Minimizing now')

states, pats, p_hist = optimize_dual_ris(states, "min", ris_list, sdr_rx,
                                         generate_pattern=generate_pattern,
                                         send_pattern=send_pattern,
                                         measure_power=measure_power,
                                         group_len=group_len,
                                         num_rows=num_rows,
                                         num_cols=num_cols,
                                         change_period=change_period)

save_ris_patterns(pats, pos, path=FILE_PATH_MIN)

# Save or plot power history
plt.figure()
plt.plot(p_hist, marker='o')
plt.xlabel("Group index")
plt.ylabel("Min Power (dBFS)")
plt.title("Power evolution during RIS optimization")
plt.grid()
plt.show()


for ris in ris_list:     
    try:
        ris.close()
    except Exception as e:
        print(f"Couldn’t close {ris.port}: {e}")
