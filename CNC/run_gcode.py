import serial
import time

def main():
    port = 'COM4' #Set this to your actual COM port!
    baudrate = 115200  # Common baudrate for 3D printers, adjust if needed

    try:
        # Open the serial connection
        ser = serial.Serial(port, baudrate, timeout=2)
        print(f"Connected to 3D printer on {port}")

        # Allow the printer to initialize
        time.sleep(2)

        ser.write(b'G0 Z10\n')
        ser.write(b'G28\n')

        with open("run.gcode", "r") as file:
            
            for line in file:
                ser.write(b'{line}')
                #print(ser.readline().decode('utf-8').strip())
                print(line)

    except serial.SerialException as e:
        print(f"Error: {e}")

    finally:
        # Close the connection
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()

