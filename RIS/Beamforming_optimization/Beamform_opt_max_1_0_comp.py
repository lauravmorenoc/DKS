# beamform_toggle_test.py

import serial
import time
import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===================== USER PARAMETERS =====================
ris_port = 'COM22'
baudrate = 115200
reads_per_check = 20
group_size = 16
change_period = 1  # seconds
num_rows = 16
num_cols = 16
rx_lo = 5.3e9
sample_rate = 2e6
NumSamples = 50000
rx_gain = 0
tx_gain = -50
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

# ---- MAIN SCRIPT ----
ris = ris_init(ris_port, baudrate)
state = [0] * (num_rows * num_cols)
send_pattern(ris, generate_pattern(state))

sdr_tx = adi.ad9361(uri='usb:1.7.5')
sdr_rx = adi.ad9361(uri='usb:1.8.5')
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

group_side = int(np.sqrt(group_size))
power_history = []

current_power = measure_power(sdr_rx, NumSamples, reads_per_check)
power_history.append(current_power)

for row in range(0, num_rows, group_side):
    for col in range(0, num_cols, group_side):
        print(f"\nChecking group at ({row}, {col})")

        indices = []
        for r in range(group_side):
            for c in range(group_side):
                i = (row + r) * num_cols + (col + c)
                indices.append(i)

        for idx in indices:
            state[idx] ^= 1

        send_pattern(ris, generate_pattern(state))
        new_power = measure_power(sdr_rx, NumSamples, reads_per_check)

        print(f"Old power: {current_power:.2f} dBFS, New power: {new_power:.2f} dBFS")

        if new_power >= current_power:
            print("→ Change accepted.")
            current_power = new_power
        else:
            print("→ Change reverted.")
            for idx in indices:
                state[idx] ^= 1
            send_pattern(ris, generate_pattern(state))

        power_history.append(current_power)
        time.sleep(change_period)

plt.figure()
plt.plot(power_history, marker='o')
plt.xlabel("Group index")
plt.ylabel("Max Power (dBFS)")
plt.title("Power evolution during RIS optimization")
plt.grid()
plt.show(block=False)


#============================= PRINT PATTERN IMAGE =============================
final_matrix = np.array(state).reshape((16, 16))
fig3 = plt.figure(figsize=(6, 6))
ax3 = fig3.add_subplot(111)

for i in range(16):
    for j in range(16):
        color = 'red' if final_matrix[i, j] == 1 else 'white'
        rect = plt.Rectangle([j, 15 - i], 1, 1, facecolor=color, edgecolor='black')
        ax3.add_patch(rect)

ax3.set_xlim(0, 16)
ax3.set_ylim(0, 16)
ax3.set_aspect('equal')
ax3.axis('off')

legend_elements = [Patch(facecolor='red', edgecolor='black', label='State 1'),
                   Patch(facecolor='white', edgecolor='black', label='State 0')]
ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))

plt.title("Optimized RIS Pattern")
plt.tight_layout()
plt.show(block=False)

#============================ TOGGLE RIS STATE AND MEASURE ============================

n_read = 10  # Number of cycles (each cycle = state 0 + state 1)
measures_per_state = 5
power_toggle_history = []
state_label = []

# Save the optimized power as reference
p_highest = current_power

# Create new figure for live tracking
fig_live, ax_live = plt.subplots()
fig_live.canvas.manager.set_window_title("RIS Toggle Test")

for cycle in range(n_read):
    for state_value in [0, 1]:  # 0 then 1
        pattern = '!0x' + ('0' * 64 if state_value == 0 else 'F' * 64)
        send_pattern(ris, pattern)
        print(f"\nRIS state set to: {state_value}")

        for _ in range(measures_per_state):
            p = measure_power(sdr_rx, NumSamples, reads_per_check)
            power_toggle_history.append(p)
            state_label.append(state_value)

            delta = p_highest - p
            ax_live.clear()
            ax_live.plot(power_toggle_history, marker='o', color='purple', label=f"Current RIS State: {state_value}")
            ax_live.axhline(y=p_highest, color='red', linestyle='--', label=f"Optimized Power: {p_highest:.2f} dBFS")
            ax_live.set_title(f"Live Power vs. RIS State — ΔPower: {delta:.2f} dB")
            ax_live.set_xlabel("Measurement Index")
            ax_live.set_ylabel("Power (dBFS)")
            ax_live.grid(True)
            ax_live.legend(loc="lower left")
            plt.pause(0.1)

plt.ioff()
plt.show()

