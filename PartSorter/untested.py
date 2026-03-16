import base64
import os
import time
import tkinter as tk

import sys
import nxt
import nxt.locator
import nxt.motor
import nxt.sensor
import nxt.sensor.generic

import cv2

from detector import PartSorterDetector


TARGET_FPS = 8
PROCESS_INTERVAL_MS = int(1000 / TARGET_FPS)
DEBUG_PRINT_EVERY = 30
DEBUG_PRINT_LINES = 22
DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_debug.log")


class ScannerApp:


    def bumper(sensor):
        def bumpy():
            while not sensor.get_sample():
                pass
            return True
        return bumpy

    def prep():
        global brick
        global motor_shoulder
        global motor_elbow
        global touch_shoulder
        global touch_elbow
        try:
            brick = nxt.locator.find()
        except nxt.locator.BrickNotFoundError:
            print("---\n<<< Did you remember to turn the brick on? >>>\n---")
            if sys.flags.interactive:
                return
            else:
                sys.exit(0)
        motor_shoulder = brick.get_motor(nxt.motor.Port.A)
        motor_elbow = brick.get_motor(nxt.motor.Port.B)
        touch_shoulder = brick.get_sensor(nxt.sensor.Port.S1, nxt.sensor.generic.Touch)
        touch_elbow = brick.get_sensor(nxt.sensor.Port.S2, nxt.sensor.generic.Touch)

    def cleanup():
        motor_shoulder.idle()
        motor_elbow.idle()
        brick.close()

    def home():
        #try:
            print("home")
            motor_shoulder.turn(-15, 360, stop_turn = ScannerApp.bumper(touch_shoulder))
            motor_elbow.turn(15, 360, stop_turn = ScannerApp.bumper(touch_elbow))
            #time.sleep(1)
        #except:
        #    return

    def push_left():
        #try:
            print("go_left")
            motor_shoulder.turn(30, 150) 
            motor_elbow.turn(-30, 150)
            #time.sleep(1)
       # except:
        #    ScannerApp.home()
        #    return

    def push_right():
        #try:
            print("go_right")
            motor_elbow.turn(-30, 250)
            motor_shoulder.turn(30, 100)
            motor_elbow.turn(5, 90)
            motor_shoulder.turn(-30, 90) 
            #time.sleep(1)
        #except:
         #   ScannerApp.home()
         #   return

    def push_off():
        #try:
            print("go_out")
            motor_elbow.turn(-30, 250)
            motor_shoulder.turn(30, 110) 
            motor_elbow.turn(30, 40)
            motor_shoulder.turn(-70, 50)
            #time.sleep(1)
        #except:
        #    ScannerApp.home()
         #   return

    def __init__(self) -> None:
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.detector = PartSorterDetector()
        self.show_debug_text = False
        self.debug_tick = 0
        self.running = True
        self.app_open = True
        self.debug_log_file = None

        self.window = tk.Tk()
        self.window.title("PartSorter Webcam Scanner")

        view_frame = tk.Frame(self.window)
        view_frame.pack(padx=12, pady=8)

        self.camera_label = tk.Label(view_frame, text="Camera Feed", compound=tk.TOP)
        self.camera_label.grid(row=0, column=0, padx=6)
        self.debug_label = tk.Label(view_frame, text="Detection Debug", compound=tk.TOP)
        self.debug_label.grid(row=0, column=1, padx=6)

        controls = tk.Frame(self.window)
        controls.pack(padx=12, pady=8, fill=tk.X)

        self.debug_btn = tk.Button(controls, text="Show debug overlay", command=self.toggle_debug_overlay)
        self.debug_btn.pack(side=tk.LEFT, padx=8)
        self.min_size_var = tk.IntVar(value=int(self.detector.min_shape_area))
        self.min_size_label = tk.Label(controls, text=f"Min object size: {self.min_size_var.get()}")
        self.min_size_label.pack(side=tk.LEFT, padx=(16, 6))

        self.on_min_size_change(1200)

        self.min_size_var = tk.Scale(
            controls,
            from_=40,
            to=1200,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.min_size_var,
            command=self.on_min_size_change,
            length=220,
        )
        self.min_size_var.pack(side=tk.LEFT, padx=4)

        self.window.protocol("WM_DELETE_WINDOW", self.stop_app)


    def run(self) -> None:
        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            self.window.after(PROCESS_INTERVAL_MS, self.update_feed)
        else:
            self.running = False
            self._log_debug("Cannot open camera")
        ScannerApp.prep()
        ScannerApp.home()
        self.window.mainloop()

    def update_feed(self) -> None:
        if not self.app_open or not self.running:
            return
        if self.camera is None or not self.camera.isOpened():
            return
        ret, frame = self.camera.read()
        if not ret:
            self._log_debug("Can't receive frame. Exiting...")
            self.stop_scanning()
            return

        display_frame, debug_mask, debug_lines, action = self.detector.process(frame)


        if self.show_debug_text:
            for idx, msg in enumerate(debug_lines[:DEBUG_PRINT_LINES]):
                cv2.putText(
                    display_frame,
                    msg,
                    (10, 18 + 18 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        status = f"Debug: {'ON' if self.show_debug_text else 'OFF'}"
        cv2.putText(
            display_frame,
            status,
            (10, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0) if self.show_debug_text else (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

        self.debug_tick += 1
        if self.debug_tick % DEBUG_PRINT_EVERY == 0:
            self._log_debug(" | ".join(debug_lines))

        self._set_tk_image(self.camera_label, display_frame)
        self._set_tk_image(self.debug_label, debug_mask if debug_mask is not None else display_frame)
        
        if action is not None:
            print(action)
            match action:
                case "left":
                    ScannerApp.push_left()
                case "right":
                    ScannerApp.push_right()
                case "outward":
                    ScannerApp.push_off()
            ScannerApp.home()
        if self.app_open:
            self.window.after(PROCESS_INTERVAL_MS, self.update_feed)

    def stop_scanning(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.debug_log_file is not None:
            try:
                self.debug_log_file.close()
            except Exception:
                pass
            self.debug_log_file = None

    def stop_app(self) -> None:
        self.app_open = False
        self.running = False
        self.stop_scanning()
        self.window.destroy()
        ScannerApp.cleanup()

    def toggle_debug_overlay(self) -> None:
        self.show_debug_text = not self.show_debug_text
        self.debug_btn.config(text="Hide debug overlay" if self.show_debug_text else "Show debug overlay")

    def on_min_size_change(self, value: str) -> None:
        min_area = max(1, int(float(value)))
        self.detector.set_min_object_area(min_area)
        self.min_size_label.config(text=f"Min object size: {min_area}")

    def _open_debug_log(self) -> None:
        if self.debug_log_file is not None:
            return
        try:
            self.debug_log_file = open(DEBUG_LOG_PATH, "a", encoding="utf-8")
        except Exception:
            self.debug_log_file = None

    def _log_debug(self, line: str) -> None:
        self._open_debug_log()
        if self.debug_log_file is None:
            return
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.debug_log_file.write(f"[{timestamp}] {line}\n")
            self.debug_log_file.flush()
        except Exception:
            pass

    @staticmethod
    def _set_tk_image(target_label: tk.Label, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".png", rgb)
        if not ok:
            return
        photo = tk.PhotoImage(data=base64.b64encode(buffer).decode("ascii"), format="png")
        target_label.config(image=photo)
        target_label.image = photo


def main() -> None:
    app = ScannerApp()
    app.run()


if __name__ == "__main__":
    main()
