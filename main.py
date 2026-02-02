import serial
import time


import math

class Vector2:
    def __init__(self, x, y):
        self.x : float = x
        self.y : float = y
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other : 'Vector2'):
        return Vector2(self.x * other.x, self.y * other.y)

    def __rmul__(self, other : 'Vector2'):
        return Vector2(other.x * self.x, other.y * self.y)

    def __mul__(self, other : float):
        return Vector2(self.x * other, self.y * other)

    def __rmul__(self, other : float):
        return Vector2(other * self.x, other * self.y)
    
    def __truediv__(self, other : float):
        return Vector2(self.x / other, self.y / other)
    
    # def __truediv__(self, other: 'Vector2'):
    #     return Vector2(self.x / other.x, self.y / other.y)

    def magnitude(self):
        return math.sqrt((self.x * self.x) + (self.y * self.y))
    
    def normalized(self):
        return self / self.magnitude()
    
    def rotated(self, angle : float) -> 'Vector2':
        return Vector2(
                (self.x * math.cos(angle)) - (self.y * math.sin(angle)),
                (self.x * math.sin(angle)) + (self.y * math.cos(angle)))
    
    def angle(self) -> float: # (1, 0) is 0 degrees
        return math.atan2(self.y, self.x)
        # if (self.magnitude() == 0):
        #     return 0.0
        # adotx = self.x # (1 * x) + (0 * y)
        # costheta = adotx / (self.magnitude())
        # return math.acos(costheta)


    def angle_between(self, other : 'Vector2') -> float:
        return self.angle() - other.angle()


class Limb:

    def __init__(self, length : float, range : tuple[float,float], last : 'Limb' = None, next : 'Limb' = None):
        self.length : float = length # length of the limb, generic units
        self.range : tuple[float,float] = range # minimum and maximum allowable angle of joint in radians (located at start of limb, not end)
        self.last : 'Limb' = last # the limb prior to this
        self.has_last : bool = last != None
        self.next : 'Limb' = next # the limb after this
        self.has_next : bool = next != None
        if self.has_last:
            self.inner_position : Vector2 = last.outer_position
        else:
            self.inner_position : Vector2 = Vector2(0,0) # assume root, start at origin
        self.outer_position : Vector2 = self.inner_position + Vector2(length, 0) # NOTE: doesn't check valid angles when declaring
        self.angle : float = 0.0 # "upwards"

    
    def set_next(self, next: 'Limb'):
        self.next = next
        self.has_next = next != None




class FABRIK2D:

    limbs : list[Limb] = []
    accuracy  = 0 # float. the distance in generic units that is acceptable between the target and the real position

    def __init__(self, lengths : list[float], ranges : list[tuple[float,float]], accuracy : float):
        self.accuracy = accuracy
        last : Limb = None
        for i in range(len(ranges)):
            l = lengths[i]
            r = ranges[i]
            r_converted = (math.radians(r[0]), math.radians(r[1]))
            current : Limb = Limb(l, r_converted, last)
            self.limbs.append(current)
            if (last != None):
                last.set_next(current)
            last = current


    def FABRIK(self, target : Vector2, max_iterations = 10000) -> list[float]:
        real_end_pos : Vector2 = self.limbs[len(self.limbs) - 1].outer_position
        iterations = 0

        while ((real_end_pos - target).magnitude() > self.accuracy and iterations < max_iterations):
            self.backwards_pass(target)
            self.forward_pass()
            real_end_pos = self.limbs[len(self.limbs) - 1].outer_position
            iterations += 1
        
        output_list = []
        for l in self.limbs:
            output_list.append(math.floor(math.degrees(l.angle)))
        return output_list

    def backwards_pass(self, goal_pos : Vector2):
        current_limb = self.limbs[len(self.limbs) - 1]
        current_limb.outer_position = goal_pos
        while True:
            if current_limb.has_last:
                super_inner_point = current_limb.last.inner_position
            else:
                super_inner_point = Vector2(0,0)

            root_angle = (current_limb.inner_position - super_inner_point).angle()
            diff_angle_raw = (current_limb.outer_position - current_limb.inner_position).angle() - root_angle
            diff_angle = min(max(diff_angle_raw, current_limb.range[0]), current_limb.range[1])

            current_limb.angle = root_angle + diff_angle

            current_limb.inner_position = current_limb.outer_position + Vector2(current_limb.length, 0).rotated(current_limb.angle + math.pi)

            if current_limb.has_last:
                current_limb.last.outer_position = current_limb.inner_position
                current_limb = current_limb.last
            else:
                break

    def forward_pass(self):
        root_angle : float = 0.0
        current_limb : Limb = self.limbs[0]
        current_limb.inner_position = Vector2(0,0)
        while True:

            diff_angle_raw = min(max((current_limb.outer_position - current_limb.inner_position).angle() - root_angle, -(math.pi)), math.pi)
            diff_angle = min(max(diff_angle_raw, current_limb.range[0]), current_limb.range[1])

            current_limb.angle = root_angle + diff_angle

            current_limb.outer_position = current_limb.inner_position + Vector2(current_limb.length, 0).rotated(current_limb.angle)

            temp = root_angle            
            root_angle = current_limb.angle

            current_limb.angle = current_limb.angle - temp



            if current_limb.has_next:
                current_limb.next.inner_position = current_limb.outer_position
                current_limb = current_limb.next
            else:
                break
    

#print(FABRIK2D([30,30],[(70,290),(70,290)], 0.1).FABRIK(Vector2(30,30)))

#print(FABRIK2D([30,30],[(70,290),(70,290)], 0.1).FABRIK(Vector2(60,0)) == [90, 0])
#print(FABRIK2D([30,30],[(70,290),(70,290)], 0.1).FABRIK(Vector2(30,30)) == [90, 90])

def cast_range_to_range(start_min, start_max, end_min, end_max, value) -> int:
	return math.floor(end_min + ((end_max - end_min) / (start_max - start_min)) * (value - start_min))

def send_command(first, second, active, wait_time):
	ser.write(('{' + str(first) + ',' + str(second) +',' + str(active) + '}').encode())
	ser.flush()
	time.sleep(wait_time)
	print( ser.readline().decode('utf-8').strip() )

def place_dot(pos, fabrik, ser):
	angles = fabrik.FABRIK(pos)
	val_first = cast_range_to_range(0, 130, 380, 85, angles[0])
	val_second = cast_range_to_range(50, 210, 75, 470, angles[1])
	print("GOTO:", str(val_first), " ", str(val_second))
	send_command(val_first, val_second, 1, 0.5)
	send_command(val_first, val_second, 2, 0.1)
	send_command(val_first, val_second, 1, 0.1)

def place_line(pos1, pos2, fabrik, ser, dashed = False, segments = 10):
	delta = pos2 - pos1
	delta = delta / segments
	curpos = pos1
	
	angles1 = fabrik.FABRIK(pos1)
	val_first_first = cast_range_to_range(0, 130, 380, 85, angles1[0])
	val_first_second = cast_range_to_range(50, 210, 75, 470, angles1[1])
	send_command(val_first_first, val_first_second, 1, 0.5)
	send_command(val_first_first, val_first_second, 2, 0.5)

	for i in range(0, segments - 1):
		curpos += delta
		curangle = fabrik.FABRIK(curpos)
		cur_val_first = cast_range_to_range(0, 130, 380, 85, curangle[0])
		cur_val_second = cast_range_to_range(50, 210, 75, 470, curangle[1])
		if dashed:
			up_state = (i / 2 % 2) + 1
		else:
			up_state = 2	

		send_command(cur_val_first, cur_val_second, up_state, 0.1)

	angles2 = fabrik.FABRIK(pos2)
	val_second_first = cast_range_to_range(0, 130, 380, 85, angles2[0])
	val_second_second = cast_range_to_range(50, 210, 75, 470, angles2[1])
	send_command(val_second_first, val_second_second, 2, 0.5)
	send_command(val_second_first, val_second_second, 1, 0.5)


def place_square(point1, point2, fabrik, ser):
	place_line(Vector2(point1.x, point1.y), Vector2(point1.x, point2.y), fabrik, ser)
	place_line(Vector2(point1.x, point2.y), Vector2(point2.x, point2.y), fabrik, ser)
	place_line(Vector2(point2.x, point2.y), Vector2(point2.x, point1.y), fabrik, ser)
	place_line(Vector2(point2.x, point1.y), Vector2(point1.x, point1.y), fabrik, ser)
	place_line(Vector2(point1.x, point1.y), Vector2(point1.x, point2.y), fabrik, ser)

def place_circle(point1, radius, fabrik, ser, segments):
	
	angles1 = fabrik.FABRIK(point1 + (Vector2(math.cos(0), math.sin(0)) * radius))
	val_first_first = cast_range_to_range(0, 130, 380, 85, angles1[0])
	val_first_second = cast_range_to_range(50, 210, 75, 470, angles1[1])
	send_command(val_first_first, val_first_second, 1, 0.5)
	send_command(val_first_first, val_first_second, 2, 0.5)


	for i in range(1, segments + 2):
		angle = (i / segments) * math.pi * 2.0
		curpos = point1 + (Vector2(math.cos(angle), math.sin(angle)) * radius)
		fabrik_angles = fabrik.FABRIK(curpos)
		val_first = cast_range_to_range(0, 130, 380, 85, fabrik_angles[0])
		val_second = cast_range_to_range(50, 210, 75, 470, fabrik_angles[1])
		send_command(val_first, val_second, 2, 0.1)
	send_command(val_first_first, val_first_second, 2, 0.5)
	send_command(val_first_first, val_first_second, 1, 0.5)

ser = serial.Serial('COM12', 115200, timeout=1)
print( ser.readline().decode('utf-8').strip() )
number = 100
diff = 100
count = 0

# first arm
#380 val to 0 deg
#75 val to 130 deg

# second arm
# 75 val to 50 deg
# 470 val to 210 deg 


# stable zone
# 25, 125
# 50, 75


fabrik = FABRIK2D([86,112],[(0, 120),(55, 210)], 0.01)

exit_flag = False


print( ser.readline().decode('utf-8').strip() )

while not exit_flag:
	print("0: draw point")
	print("1: draw line")
	print("2: draw square")
	print("3: draw circle")
	print("4: exit")
	input_val = input("Please Input Command:")

	match int(input_val):
		case 0:
			x = float(input("X coordinate:"))
			y = float(input("Y coordinate:"))
			place_dot(Vector2(x, y), fabrik, ser)
		case 1:
			x1 = float(input("first X coordinate:"))
			y1 = float(input("first Y coordinate:"))
			x2 = float(input("second X coordinate:"))
			y2 = float(input("second Y coordinate:"))
			dotted = (input("dotted line? (Y/N)") == "Y")
			segments = int(input("Number of segments:"))
			place_line(Vector2(x1, y1), Vector2(x2, y2), fabrik, ser, dotted, segments)
		case 2:
			x1 = float(input("first X coordinate:"))
			y1 = float(input("first Y coordinate:"))
			x2 = float(input("second X coordinate:"))
			y2 = float(input("second Y coordinate:"))
			place_square(Vector2(x1, y1), Vector2(x2, y2), fabrik, ser)
		case 3:
			x = float(input("centre X coordinate:"))
			y = float(input("centre Y coordinate:"))
			radius = float(input ("enter radius:"))
			sides = int(input ("enter number of sides:"))
			place_circle(Vector2(x, y), radius, fabrik, ser, sides)
		case 4:
			print("Exiting...")
			exit_flag = True
		case _:
			print("input not understood")
			
	

	#ser.write(('{100,100,1}').encode())
	#while True:
	#	place_dot(Vector2(0 - (count * 25), 125), fabrik, ser)
	#	count += 1
	#	time.sleep(0.1)
#place_line(Vector2(-50, 100), Vector2(0, 75), fabrik, ser, True, 10)
#place_line(Vector2(0, 75), Vector2(-25, 50), fabrik, ser, True, 10)
#place_line(Vector2(-25, 50), Vector2(-50, 100), fabrik, ser, True, 10)

#place_square(Vector2(-50,100), Vector2(-25, 75), fabrik, ser)
#place_circle(Vector2(-38.5, 87.5), 12.5, fabrik, ser)