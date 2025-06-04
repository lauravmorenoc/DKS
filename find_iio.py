import subprocess
import re

def find_usb_contexts_with_serials():
    try:
        # Run iio_info -s and capture output
        result = subprocess.run(["iio_info", "-s"], capture_output=True, text=True)
        output = result.stdout

        print("Raw output:\n", output)

        # Split output into blocks for each context
        context_blocks = output.strip().split("IIO context found:")

        usb_info = []

        for block in context_blocks:
            if "usb:" in block:
                # Extract USB URI
                usb_match = re.search(r'uri:\s*(usb:[\d\.]+)', block)
                # Extract serial number
                serial_match = re.search(r'serial:\s*(\S+)', block)

                if usb_match and serial_match:
                    usb_uri = usb_match.group(1)
                    serial = serial_match.group(1)
                    usb_info.append((usb_uri, serial))
                    print(f"Found USB context: {usb_uri}, Serial: {serial}")

        return usb_info

    except Exception as e:
        print(f"Error finding USB contexts and serials: {e}")
        return []

# Run it
find_usb_contexts_with_serials()