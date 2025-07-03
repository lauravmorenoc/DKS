import serial, time, re, numpy as np, adi, matplotlib.pyplot as plt

# ───────────── user settings ────────────────────────────────────
pos          = 7                # which line (1-9) to use
file_max     = "optimized_max_ris_pattern_hex.txt"
file_min     = "optimized_min_ris_pattern_hex.txt"
ris_ports    = ['COM18', 'COM19']  # board A, board B
baudrate     = 115200
rx_lo        = 5.3e9
tx_gain      = 0
rx_gain      = 30
sample_rate  = 2.6e6
NumSamples   = 300_000
reads_per_check = 3
cycles_per_mode   = 2
measures_per_state= 10
# ───────────────────────────────────────────────────────────────

# ---------- helpers --------------------------------------------
def read_two_patterns(path, wanted_pos):
    """Return (patA, patB) for the requested [pos] line."""
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*(?P<a>!0x[0-9a-fA-F]+)\s+'
                         r'(?P<b>!0x[0-9a-fA-F]+)\s+\['
                         r'(?P<idx>\d+)\]', line)
            if m and int(m.group('idx')) == wanted_pos:
                return m.group('a'), m.group('b')
    raise ValueError(f"Position {wanted_pos} not found in {path}")

def measure_power(dev):
    vals=[]
    for _ in range(reads_per_check):
        d = dev.rx()[0].astype(np.float32)/(2**14)
        vals.append(20*np.log10(np.mean(np.abs(d)**2)+1e-12))
    return np.mean(vals)

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

        if not uri_tx or not uri_rx:
            raise RuntimeError(f"Could not find both required SDRs. Found TX={uri_tx}, RX={uri_rx}")

        return uri_tx, uri_rx

    except Exception as e:
        print(f"Error during SDR URI detection: {e}")
        raise

# ---------- load patterns --------------------------------------
maxA, maxB = read_two_patterns(file_max, pos)
minA, minB = read_two_patterns(file_min, pos)

baseline = ['!0x'+'0'*64, '!0x'+'F'*64]   # all-0 vs all-1
optimised= [minA+' '+minB, maxA+' '+maxB] # we’ll split later
modes    = [baseline, optimised]
mode_names = ["baseline", "optimised"]

# ---------- open both RIS boards -------------------------------
def ris_init(port):
    h = serial.Serial(port, baudrate, timeout=1)
    h.write(b'!Reset\n'); time.sleep(1)
    while h.in_waiting: _ = h.readline()
    return h
ris_list = [ris_init(p) for p in ris_ports]

# ---------- SDR for power measurement --------------------------
uri_tx, uri_rx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)
sdr_rx = adi.ad9361(uri=uri_rx)
sdr_tx.sample_rate = sdr_rx.sample_rate = int(sample_rate)
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

# ---------- live Δ-power loop ----------------------------------
plt.ion(); fig,ax = plt.subplots()
x_b, x_o = [], []
d_b, d_o = [], []
i_mode=0; cycle=0

try:
    while True:
        high, low = modes[i_mode]          # strings
        pats_hi = high.split() if ' ' in high else [high, high]
        pats_lo = low.split()  if ' ' in low  else [low,  low]

        print(f"\n--- {mode_names[i_mode]} mode, cycles {cycle+1}-{cycle+cycles_per_mode} ---")
        for _ in range(cycles_per_mode):
            for pair,label in ((pats_lo,"0"),(pats_hi,"1")):
                # send current pair to both boards
                for h,pat in zip(ris_list, pair):
                    h.write((pat+'\n').encode()); time.sleep(0.05)
                    if h.in_waiting: h.readline()

                vals=[measure_power(sdr_rx) for _ in range(measures_per_state)]
                print(f" state {label}: {np.mean(vals):.2f} dBFS")

                target_x = x_b if i_mode==0 else x_o
                target_d = d_b if i_mode==0 else d_o
                start=len(target_d)+1
                target_x.extend(range(start, start+len(vals)))
                target_d.extend(vals)

        ax.clear()
        ax.plot(x_b,d_b,'b.-',label='baseline')
        ax.plot(x_o,d_o,'g.-',label='optimised')
        ax.set_xlabel("Sample"); ax.set_ylabel("Power (dBFS)")
        ax.legend(); ax.grid(True); plt.pause(0.1)

        i_mode ^= 1; cycle += cycles_per_mode

except KeyboardInterrupt:
    pass

plt.ioff(); plt.show()
for h in ris_list: h.close()
