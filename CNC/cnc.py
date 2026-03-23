#Sample movement to ensure communication with the 3D printer
import serial
import time

def main():
    port = 'COM3' #Set this to your actual COM port!
    baudrate = 115200  # Common baudrate for 3D printers, adjust if needed

    try:
        # Open the serial connection
        ser = serial.Serial(port, baudrate, timeout=2)
        print(f"Connected to 3D printer on {port}")

        # Allow the printer to initialize
        time.sleep(2)

        # Send GCode to raise the tip by 10mm
        ser.write(b'G91\n')  # Set to relative positioning
        ser.write(b'G0 Z10\n')  # Move up by 10mm
        time.sleep(1)

        # Send GCode to move along the positive X axis by 10mm
        ser.write(b'G0 X10\n')  # Move 10mm in X direction
        time.sleep(1)

        # Send GCode to lower the tip by 10mm
        ser.write(b'G0 Z-10\n')  # Move down by 10mm
        time.sleep(1)

        # Set back to absolute positioning (optional, for safety)
        ser.write(b'G90\n')

        print("Commands sent successfully.")

    except serial.SerialException as e:
        print(f"Error: {e}")

    finally:
        # Close the connection
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()