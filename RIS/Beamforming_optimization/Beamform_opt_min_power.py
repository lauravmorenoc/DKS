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
change_period= 1 # seconds
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
state = [0] * (num_rows * num_cols)  # initial state: all OFF
send_pattern(ris, generate_pattern(state))

# Configure SDR
sdr_tx = adi.ad9361(uri='usb:1.7.5')
sdr_rx = adi.ad9361(uri='usb:1.8.5')
#sdr_tx=sdr_rx=adi.ad9361(uri='usb:1.5.5')
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
plt.show(block=False)

# Saving in text file
hex_pattern = generate_pattern(state)
with open("optimized_min_ris_pattern_hex.txt", "w") as f:
    f.write(hex_pattern)

#============================= PRINT PATTERN IMAGE =============================

# Reshape 256-element state into 16×16 matrix
final_matrix = np.array(state).reshape((16, 16))

# Create a new figure just for the matrix
fig3 = plt.figure(figsize=(6, 6))  # Use a unique figure name
ax3 = fig3.add_subplot(111)

# Draw the matrix
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


#============================ LIVE POWER MEASUREMENT ============================

number_of_reads = 1000  # Total live readings
window_size = 50       # Max number of readings in window
power_history_live = []

print(f"Starting live power tracking with max power = {current_power:.2f} dBFS")

# Create new figure for live plot
fig_live, ax_live = plt.subplots()
fig_live.canvas.manager.set_window_title("Live Power Plot")

for i in range(number_of_reads):
    live_power = measure_power(sdr_rx, NumSamples, reads_per_check)
    power_history_live.append(live_power)

    if len(power_history_live) > window_size:
        power_history_live = power_history_live[-window_size:]

    ax_live.clear()  # ONLY clear this figure's axes, not others
    ax_live.plot(power_history_live, marker='o', color='orange', label="Live Power Readings")
    ax_live.axhline(y=current_power, color='red', linestyle='--',
                    label=f"Max Power: {current_power:.2f} dBFS")

    delta = current_power - live_power
    ax_live.set_title(f"Live Tracking — ΔPower: {delta:.2f} dB")
    ax_live.set_xlabel("Read Index (last 50)")
    ax_live.set_ylabel("Power (dBFS)")
    ax_live.grid(True)
    ax_live.legend(loc="lower left")
    plt.pause(0.25)

plt.ioff()
plt.show()
