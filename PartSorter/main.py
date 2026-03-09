import time
import sys
import nxt
import nxt.locator
import nxt.motor
import nxt.sensor
import nxt.sensor.generic
import cv2

# Open the default camera (0 = first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open camera")
	exit()





while True:
	# Read a frame
	ret, frame = cap.read()

	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")


	# Show the frame
	cv2.imshow('Camera Feed', frame)

	# Exit on pressing 'q'
	if cv2.waitKey(1) == ord('q'):
		break

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
	motor_elbow.turn(15, 360, stop_turn = bumper(touch_elbow))
	motor_shoulder.turn(-15, 360, stop_turn = bumper(touch_shoulder))

def push_left():
	motor_shoulder.turn(30, 150) 
	motor_elbow.turn(-30, 150)
def push_right():
	motor_elbow.turn(-30, 250)
	motor_shoulder.turn(30, 100)
	motor_elbow.turn(5, 90)
	motor_shoulder.turn(-30, 90) 
def push_off():
	motor_elbow.turn(-30, 250)
	motor_shoulder.turn(30, 110) 
	motor_elbow.turn(30, 40)
	motor_shoulder.turn(-70, 50) 
prep()
home()
time.sleep(1)
"""
push_right()
time.sleep(1)
home()
time.sleep(1)
push_left()
time.sleep(1)
home()
time.sleep(1)
push_off()
time.sleep(1)


home()
time.sleep(1)
"""
cleanup() # for when you're done.