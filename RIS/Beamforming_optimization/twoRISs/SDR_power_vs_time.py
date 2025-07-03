import collections as _cl
import matplotlib.pyplot as plt
import numpy as np
import adi, time

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

            if serial.endswith("0a"):
                uri_tx = uri
                print("Assigned as Tx")
            elif serial.endswith("9e"):
                uri_rx = uri
                print("Assigned as Rx")

     #   if not uri_tx or not uri_rx:
       #     raise RuntimeError(f"Could not find both required SDRs. Found TX={uri_tx}, RX={uri_rx}")

        return uri_tx, uri_rx

    except Exception as e:
        print(f"Error during SDR URI detection: {e}")
        raise

def read_power_dbfs(dev, n_avg):
    """Return mean power in dBFS averaged over n_avg buffers."""
    vals = []
    for _ in range(n_avg):
        data = dev.rx()[0]                  # complex I/Q
        p_lin = np.mean(np.abs(data)**2)    # linear power
        vals.append(20*np.log10(p_lin / (2**14)))   # 14-bit scaling
    return np.mean(vals)

# ───────────────────────── Radio setup  ──────────────────────────
samp_rate   = 2e6
NumSamples  = 2**12
rx_lo       = 5.3e9
rx_mode     = "manual"
rx_gain    = 0
reads_per_check = 3            # how many buffers we average for 1 “power” point
window_pts  = 100              # plot only the last N measurements
num_scans   = 10000
tx_gain = 0

# === Initialize SDR ===
uri_tx, uri_rx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)
sdr_rx = adi.ad9361(uri=uri_rx)
sdr_tx.sample_rate = sdr_rx.sample_rate = int(samp_rate)
sdr_tx.rx_rf_bandwidth = sdr_rx.rx_rf_bandwidth = int(3 * 200e3)
sdr_tx.rx_lo = sdr_rx.rx_lo = int(rx_lo)
sdr_tx.gain_control_mode = sdr_rx.gain_control_mode = "manual"
sdr_tx.rx_hardwaregain_chan0 = sdr_rx.rx_hardwaregain_chan0 = rx_gain
sdr_tx.rx_buffer_size = sdr_rx.rx_buffer_size = NumSamples
sdr_tx.tx_rf_bandwidth = int(3 * 200e3)
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


# ─────────────────────────── live plot loop ─────────────────────────────────
plt.ion()
fig, ax = plt.subplots()
ydata   = _cl.deque(maxlen=window_pts)      # rolling window
xdata   = _cl.deque(maxlen=window_pts)
#ydata=[]
#xdata=[]
ln,     = ax.plot([], [], marker='o')       # initialise empty Line2D

ax.set_ylim(-90, -40)                        # adjust to your signal level
ax.set_xlabel("Sample #")
ax.set_ylabel("Power  [dBFS]")
ax.grid(True)

for i in range(num_scans):
    p_dbfs = read_power_dbfs(sdr_rx, reads_per_check)
    ydata.append(p_dbfs)
    xdata.append(i)

    ln.set_data(xdata, ydata)
    ax.set_xlim(max(0, i-window_pts), i)    # keep the last 100 points visible
    ax.set_title(f"Live Power")

    fig.canvas.draw(); fig.canvas.flush_events()
    time.sleep(0.01)                        # ~10 fps; adjust as you like

plt.ioff()
plt.show()

sdr_rx.rx_destroy_buffer()
sdr_tx.rx_destroy_buffer()
print('\a')       # beep when done
