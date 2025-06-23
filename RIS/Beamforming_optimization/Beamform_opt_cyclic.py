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
#   group_size = 16
change_period = 1  # seconds
num_rows = 16
num_cols = 16
rx_lo = 5.3e9
sample_rate = 2.6e6
NumSamples = 300000
rx_gain = 0
tx_gain = 0
group_len   = 8      # width of a stripe
group_height = 1     # stripes are one-row tall
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
        Rx_0 = Rx_0 / (2**14)
        Rx_0 = (abs(Rx_0))**2
        peak_power = 20 * np.log10(np.mean(Rx_0))
        powers.append(peak_power)
    return np.mean(powers)

def optimize_ris(state, mode, ris, sdr_rx,
                 generate_pattern, send_pattern, measure_power,
                 group_len, num_rows, num_cols, change_period):

    power_history = []
    current_power = measure_power(sdr_rx, NumSamples, reads_per_check)
    power_history.append(current_power)

    # Walk every row (step = 1) ...
    for row in range(num_rows):
        # ... and split it into stripes of `group_len` columns (0-7, 8-15)
        for col in range(0, num_cols, group_len):
            indices = [row * num_cols + (col + offset)
                       for offset in range(group_len)]

            # ── Toggle the whole stripe ───────────────────────────────
            for idx in indices:
                state[idx] ^= 1

            send_pattern(ris, generate_pattern(state))
            new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

            # keep change only if it helps (max or min mode)
            if (mode == "max" and new_power >= current_power) or \
               (mode == "min" and new_power <= current_power):
                current_power = new_power
            else:
                for idx in indices:
                    state[idx] ^= 1    # revert
                send_pattern(ris, generate_pattern(state))

            power_history.append(current_power)
            time.sleep(change_period)

    return state, generate_pattern(state)

def plot_all_patterns(pattern_list):
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    axs = axs.flatten()

    for idx, pattern in enumerate(pattern_list):
        state_bin = bin(int(pattern[3:], 16))[2:].zfill(256)
        matrix = np.array([int(b) for b in state_bin]).reshape((16, 16))
        ax = axs[idx]
        for i in range(16):
            for j in range(16):
                color = 'red' if matrix[i, j] == 1 else 'white'
                rect = plt.Rectangle([j, 15 - i], 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        ax.set_aspect('equal')
        ax.axis('off')
        if   idx < 3: mode_lbl = "Max"
        elif idx < 6: mode_lbl = "Min"
        else: mode_lbl = "Max" if (idx - 6) % 2 == 0 else "Min"
        title = f"Pattern {idx+1} ({mode_lbl})"
        ax.set_title(title, fontsize=10)

    legend_elements = [Patch(facecolor='red', edgecolor='black', label='State 1'),
                       Patch(facecolor='white', edgecolor='black', label='State 0')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    plt.show()

# ========================= MAIN SCRIPT =========================

ris = ris_init(ris_port, baudrate)
state = [0] * (num_rows * num_cols)
send_pattern(ris, generate_pattern(state))

sdr_tx = adi.ad9361(uri='usb:1.8.5')
sdr_rx = adi.ad9361(uri='usb:1.7.5')
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

num_symbols = 10
sps = 16
x_symbols = np.ones(num_symbols)
samples = np.repeat(x_symbols, sps)
samples_tx = samples * (2**14)
sdr_tx.tx([samples_tx, samples_tx])

#group_side = int(np.sqrt(group_size))
all_patterns = []

# First: 3 maximizations
for i in range(3):
    print(f"\n--- MAXIMIZATION PHASE {i+1} ---")
    state, pat = optimize_ris(state, "max", ris, sdr_rx, generate_pattern, send_pattern,
                              measure_power,  group_len, num_rows, num_cols, change_period)
    all_patterns.append(pat)


# Then: 3 minimizations
for i in range(3):
    print(f"\n--- MINIMIZATION PHASE {i+1} ---")
    state, pat = optimize_ris(state, "min", ris, sdr_rx, generate_pattern, send_pattern,
                              measure_power,  group_len, num_rows, num_cols, change_period)
    all_patterns.append(pat)
'''
# Then alternate 6 more times (3 max + 3 min interleaved)
for i in range(6):
    mode = "max" if i % 2 == 0 else "min"
    print(f"\n--- ALTERNATING PHASE {i+1} ({mode.upper()}) ---")
    state, pat = optimize_ris(state, mode, ris, sdr_rx, generate_pattern, send_pattern,
                              measure_power,  group_len, num_rows, num_cols, change_period)
    all_patterns.append(pat)'''
for i in range(6):
    all_patterns.append(pat)

plot_all_patterns(all_patterns)
print("\n--- All 12 optimization steps completed ---")
