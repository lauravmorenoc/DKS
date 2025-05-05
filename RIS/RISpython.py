import serial
import time

# ===============================
# INITIALISIERUNG
# ===============================

def ris_init(port, baudrate):
    ris = serial.Serial(port, baudrate, timeout=1)
    
    # Reset RIS
    ris.write(b'!Reset\n')
    time.sleep(1)
    
    # Leere den Eingangspuffer
    while ris.in_waiting > 0:
        response = ris.readline().decode().strip()
        print(f"Response from resetting RIS: {response}")
        time.sleep(0.1)
    
    time.sleep(0.1)
    while ris.in_waiting > 0:
        _ = ris.readline()
        time.sleep(0.1)
    
    return ris

# ===============================
# SEQUENZAUSFÜHRUNG
# ===============================

def ris_sequence(ris, high, low, sequence, period, duration, sleep_time):
    elapsed = 0
    while elapsed < duration:
        for value in sequence:
            if value == 0:
                current_pattern = low
            elif value == 1:
                current_pattern = high
            else:
                print("Could not write sequence value, it must be either 0 or 1")
                continue
            
            ris.write((current_pattern + '\n').encode())
            response = ris.readline().decode().strip()
            #print(f"Response from setting a pattern: {response}")
            #print(f"Current pattern: {current_pattern}")
            
            time.sleep(period)
            elapsed += period
        
        time.sleep(sleep_time)

# ===============================
# HAUPTPROGRAMM
# ===============================

if __name__ == "__main__":
    
    all_off = '!0x0000000000000000000000000000000000000000000000000000000000000000'
    all_on  = '!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
    
    high = all_off
    low = all_on
    period = 2.5e-6  # Sekunden
    duration = 50  # Sekunden
    sleep_time = 0

    # Beispielsequenz
    mseq = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # RIS öffnen
    ris = ris_init('COM5', 115200)

    '''
    current_pattern=low
    ris.write((current_pattern + '\n').encode())
    response = ris.readline().decode().strip()
    print(f"Response from setting a pattern: {response}")'''

    # Sequenz ausführen
    ris_sequence(ris, high, low, mseq, period, duration, sleep_time)

