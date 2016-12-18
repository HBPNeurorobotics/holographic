import math
import pygame
import random

from enum import Enum
from pygame.locals import *

import sys
sys.path.append("../vsa/")
from hrr import HRR


class Space(Enum):
	LOCAL = 1
	WORLD = 2


class Shape(Enum):
	CIRCLE = 1
	RECT = 2
	HOUSE = 3
	FAN = 4


class WheelDir(Enum):
	FORWARDS = "FORWARDS"
	BACKWARDS = "BACKWARDS"
	NONE = "NONE"


Color = {
	"WHITE"  : (255, 255, 255),
	"BLACK"  : (  0,   0,   0),
	"RED"    : (255,   0,   0),
	"GREEN"  : (  0, 255,   0),
	"BLUE"   : (  0,   0, 255),
	"CYAN"   : (  0, 255, 255),
	"MAGENTA": (255,   0, 255),
	"YELLOW" : (255, 255,   0),
};


def approx_equal(a, b, tolerance):
	return abs(a - b) <= max(abs(a), abs(b)) * tolerance


class Vec2(object):
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	def __add__(self, other):
		return Vec2(self.x + other.x, self.y + other.y)

	def __sub__(self, other):
		return Vec2(self.x - other.x, self.y - other.y)

	def __mul__(self, other):
		return Vec2(self.x * other, self.y * other)

	def __neg__(self):
		return Vec2(-self.x, -self.y)

	def __str__(self):
		return "{{ {}, {} }}".format(self.x, self.y)

	@staticmethod
	def dot(a, b):
		return a.x * b.x + a.y * b.y

	def length(self):
		return math.sqrt(self.x * self.x + self.y * self.y)

	def normalized(self):
		l = self.length()
		res = Vec2(self.x / l, self.y / l)
		assert approx_equal(res.length(), 1.0, 0.000001)
		return res

	def rotated(self, angle):
		# NOTE: rotation is counterclockwise
		sa = math.sin(angle)
		ca = math.cos(angle)
		res = Vec2(self.x * ca - self.y * sa, self.x * sa + self.y * ca)
		assert approx_equal(res.length(), self.length(), 0.0001)
		return res

	def to_int_array(self):
		return [int(self.x + 0.5), int(self.y + 0.5)]


class Default(object):
	FORWARD = Vec2(x=0.0, y=1.0)
	RIGHT = Vec2(x=1.0, y=0.0)


class Rot2(object):
	# NOTE: rotation is counterclockwise
	def __init__(self, angle=0.0):
		self._angle = self.angle = angle

	def __add__(self, other):
		return Rot2(self._angle + other._angle)

	def __str__(self):
		return "{{ {} rad }}".format(self._angle)

	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, value):
		# normalize to [0, 2*pi)
		self._angle = value % (2 * math.pi)


class Transform(object):
	def __init__(self, parent=None):
		self._position = self.local_position = Vec2()
		self._orientation = self.local_orientation = Rot2();
		self._parent = self.parent = parent
		self._children = []

	def __str__(self):
		return "{{ position: {}, orientation: {}, parent: {}, len(children): {} }}".format(
				self._position, self._orientation,
				self.parent is not None, len(self._children))

	def _local_rotated_vector(self, vec):
		# rotate in local space
		return vec.rotated(self.local_orientation.angle)

	def _rotated_vector(self, vec):
		# rotate in world space
		return vec.rotated(self.orientation.angle)

	@property
	def forward(self):
		# forward direction of this transform in world space
		return self._rotated_vector(Default.FORWARD)

	@property
	def right(self):
		# right direction of this transform in world space
		return self._rotated_vector(Default.RIGHT)

	@property
	def local_position(self):
		# return position relative to parent transform
		return self._position

	@local_position.setter
	def local_position(self, value):
		# set local position from local space value
		assert isinstance(value, Vec2)
		self._position = value

	@property
	def position(self):
		# return position in world space
		if self.parent is None:
			return self._position
		else:
			return self.parent.position + self.parent._rotated_vector(self._position)

	@property
	def local_orientation(self):
		# return orientation relative to parent transform
		return self._orientation

	@local_orientation.setter
	def local_orientation(self, value):
		# set orientation from local space value
		assert isinstance(value, Rot2)
		self._orientation = value

	@property
	def orientation(self):
		# return orientation in world space
		if self.parent is None:
			return self._orientation
		else:
			return self._orientation + self.parent.orientation

	@property
	def parent(self):
		return self._parent

	@parent.setter
	def parent(self, value):
		assert value is None or isinstance(value, Transform)
		if self._parent is not None and self in self._parent._children:
			# TODO: use remove_child() method in Transform
			self._parent._children.remove(self)
		self._parent = value
		if self._parent is not None:
			# TODO: use add_child() method in Transform
			self._parent._children.append(self)

	def translate(self, translation, space=Space.LOCAL):
		if space is Space.WORLD:
			# TODO
			raise NotImplementedError
		elif space is Space.LOCAL:
			self._position += self._local_rotated_vector(translation)
		else:
			raise NotImplementedError

	def rotate(self, rotation, space=Space.LOCAL):
		# NOTE: counterclockwise
		if space is Space.WORLD:
			# TODO:
			raise NotImplementedError
		if space is Space.LOCAL:
			self._orientation += rotation
		else:
			raise NotImplementedError

	def local_transformed_vector(self, vec):
		# transform given vector from world to local space
		return (vec - self.position).rotated(-self.orientation.angle)


class Wheel(object):
	def __init__(self, velocity, wheel_dir=WheelDir.NONE):
		self.hrr = HRR(random.random())
		self.velocity = velocity
		self._value = self.value = wheel_dir

	def _dir_to_val(self, wheel_dir):
		assert isinstance(wheel_dir, WheelDir)
		if wheel_dir is WheelDir.FORWARDS:
			return 1.0
		elif wheel_dir is WheelDir.BACKWARDS:
			return -1.0
		elif wheel_dir is WheelDir.NONE:
			return 0.0
		else:
			raise NotImplementedError

	@property
	def value(self):
		return self._value * self.velocity

	@value.setter
	def value(self, value):
		self._value = self._dir_to_val(value)


class WheelSet(object):
	def __init__(self, velocity=1.0, left=WheelDir.NONE, right=WheelDir.NONE):
		self._left = Wheel(velocity, left)
		self._right = Wheel(velocity, right)

	@property
	def left(self):
		return self._left

	@left.setter
	def left(self, value):
		self._left.value = value

	@property
	def right(self):
		return self._right

	@right.setter
	def right(self, value):
		self._right.value = value


class DiffSteer(object):
	def __init__(self):
		self.axle_length = 5.0  # distance between wheels (center-to-center)

	def _rot(self, wheel_set, orientation, delta_time):
		assert isinstance(wheel_set, WheelSet)
		dvel = wheel_set.right.value - wheel_set.left.value
		if approx_equal(dvel, 0.0, 1e-6):
			return orientation
		else:
			return Rot2(dvel * delta_time / self.axle_length + orientation.angle)

	def _pos(self, wheel_set, position, orientation, delta_time):
		assert isinstance(wheel_set, WheelSet)
		res = Vec2()
		dvel = wheel_set.right.value - wheel_set.left.value
		if approx_equal(dvel, 0.0, 1e-6):
			res.x = position.x - delta_time * wheel_set.right.value * math.sin(orientation.angle)
			res.y = position.y + delta_time * wheel_set.left.value * math.cos(orientation.angle)
		else:
			svel = wheel_set.right.value + wheel_set.left.value
			radius = (self.axle_length / 2.0) * svel / dvel;
			rot = self._rot(wheel_set, orientation, delta_time)
			res.x = position.x + radius * (math.cos(rot.angle) - math.cos(orientation.angle))
			res.y = position.y + radius * (math.sin(rot.angle) - math.sin(orientation.angle))
		return res

	def move(self, wheel_set, position, rotation, delta_time):
		# NOTE: when moving forwards/backwards increase speed (hack)
		if wheel_set.left.direction == wheel_set.right.direction:
			delta_time *= 30.0
		pos = self._pos(wheel_set, position, rotation, delta_time)
		rot = self._rot(wheel_set, rotation, delta_time)
		return pos, rot


class Observation(object):
	def __init__(self, obj, distance, angle):
		self.object = obj
		# TODO: use Transform instead?
		self.distance = distance
		self.angle = angle


class VisObject(object):
	def __init__(self, shape, color):
		self.transform = Transform()

		# use random value to uniquely identify this object in HRR
		self.hrr = HRR(random.random())

		assert isinstance(shape, Shape)
		self.shape = shape
		assert Color.has_key(color)
		self.color = color

	def __str__(self):
		return "{{ shape: {} color: {} transform: {} }}".format(self.shape, self.color, self.transform)


class Sensor(VisObject):
	def __init__(self, color, field_of_view=math.pi/3.0): # pi/3 rad == 60 deg
		super(Sensor, self).__init__(Shape.FAN, color)
		self.fov = field_of_view

	def __str__(self):
		return super(Sensor, self).__str__()

	def _to_view_space(self, pos):
		cosf = Vec2.dot(Default.FORWARD, pos)
		if cosf <= 0:
			return -1.0  # behind eye
		alpha_fov = math.acos(cosf) / self.fov
		left = Vec2.dot(Default.RIGHT, pos) < 0
		pos_fov = 0.5 - alpha_fov if left else 0.5 + alpha_fov
		return pos_fov

	def read(self, objects):
		output = []
		for o in objects:
			rel_pos = self.transform.local_transformed_vector(o.transform.position)
			pos_vs = self._to_view_space(rel_pos.normalized())
			if pos_vs >= 0.0 and pos_vs <= 1.0:
				output.append((o, pos_vs))
		return output

	def read_hrr(self, objects):
		# TODO: untested! check if this actually works
		output = None
		for o in objects:
			rel_pos = self.transform.local_transformed_vector(o.transform.position)
			pos_vs = self._to_view_space(rel_pos.normalized())
			if pos_vs >= 0.0 and pos_vs <= 1.0:
				#print("{} - {}: {}".format(self.transform.local_position, o.color, pos_vs))
				if output is None:
					output = o.hrr * pos_vs
				else:
					output += o.hrr * pos_vs
		return self.hrr * output if output is not None else self.hrr


class Agent(VisObject):
	def __init__(self, color, velocity=1.0):
		super(Agent, self).__init__(Shape.HOUSE, color)
		self.controller = None
		self._sensors = []
		self.steering = DiffSteer()
		self.wheels = WheelSet(velocity=velocity)

	def __str__(self):
		return super(Agent, self).__str__()

	def add_sensor(self, sensor=None):
		if sensor is None:
			# TODO: maybe use the same color as parent?
			s = Sensor("BLACK")
			s.transform.parent = self.transform
			self._sensors.append(s)
		else:
			sensor.transform.parent = self.transform
			self._sensors.append(sensor)

	def step(self, objects, delta_time):
		self.wheels.left = WheelDir.NONE
		self.wheels.right = WheelDir.NONE
		max_similarity_left = 0.0
		max_similarity_right = 0.0
		for s in self._sensors:
			#print("S: {}".format(s))
			output = s.read(objects)
			# probe controller with sensed value for current sensor
			sensor_ctl = self.controller % s.hrr
			for obj, val in output:
				obj_ctl = sensor_ctl % obj.hrr
				actions = obj_ctl / self.wheels.left.hrr
				print ("s {} actions o {}: {}".format(s.transform.local_position, obj.color, actions))
				for a in actions:
					if a[1] > max_similarity_left:
						try:
							self.wheels.left = WheelDir(a[0])
							max_similarity_left = a[1]
						except: pass
				actions = obj_ctl / self.wheels.right.hrr
				for a in actions:
					if a[1] > max_similarity_right:
						try:
							self.wheels.right = WheelDir(a[0])
							max_similarity_right = a[1]
						except: pass
		pos, rot = self.steering.move(self.wheels,
				self.transform.local_position,
				self.transform.local_orientation, delta_time)

		# TODO remove
		#ws = WheelSet(velocity=1.0, left=WheelDir.FORWARDS, right=WheelDir.FORWARDS)
		#pos, rot = self.steering.move(ws, self.transform.local_position, self.transform.local_orientation, delta_time)

		self.transform.local_position = pos
		self.transform.local_orientation = rot


class Object(VisObject):
	def __init__(self, color):
		super(Object, self).__init__(Shape.CIRCLE, color)

	def __str__(self):
		return super(Object, self).__str__()


class World(object):
	def __init__(self, size):
		self.size = size
		self._transform = Transform()
		self.agents = []
		self.objects = []

	def add_agent(self, agent=None):
		if agent is None:
			a = Agent(color="BLACK")
			a.transform.parent = self._transform
			self.agents.append(a)
		else:
			agent.transform.parent = self._transform
			self.agents.append(agent)

	def add_object(self, obj):
		obj.transform.parent = self._transform
		self.objects.append(obj)

	def step(self, delta_time):
		for a in self.agents:
			a.step(self.objects, delta_time)


class Visualization(object):
	def __init__(self, size):
		self.screen = pygame.display.set_mode(size)

		pygame.display.set_caption("robosim")
		self.screen.fill(Color["WHITE"])
		pygame.display.flip()

	def _flip_y(self, value):
		return self.screen.get_height() - value

	def _draw(self, shape, transform, color, fan_angle=0.0):
		center = transform.position.to_int_array()
		# pygame uses (0, 0) as top-left, we want bottom-left => flip y
		center[1] = self._flip_y(center[1])

		col = Color[color]
		if shape is Shape.CIRCLE:
			r = 10
			pygame.draw.circle(self.screen, col, center, r)
		elif shape is Shape.RECT:
			l = 5.0
			v1 = (transform.position + -transform.right * l - transform.forward * l).to_int_array()
			v2 = (transform.position + -transform.right  * l + transform.forward * l).to_int_array()
			v4 = (transform.position + transform.right * l + transform.forward * l).to_int_array()
			v5 = (transform.position + transform.right * l - transform.forward * l).to_int_array()
			v1[1] = self._flip_y(v1[1])
			v2[1] = self._flip_y(v2[1])
			v4[1] = self._flip_y(v4[1])
			v5[1] = self._flip_y(v5[1])
			pygame.draw.polygon(self.screen, col, [v1, v2, v4, v5])
		elif shape is Shape.HOUSE:
			l = 5.0
			v1 = (transform.position + -transform.right * l - transform.forward * l).to_int_array()
			v2 = (transform.position + -transform.right  * l + transform.forward * l).to_int_array()
			v3 = (transform.position + transform.forward * l * 2.0).to_int_array()
			v4 = (transform.position + transform.right * l + transform.forward * l).to_int_array()
			v5 = (transform.position + transform.right * l - transform.forward * l).to_int_array()
			v1[1] = self._flip_y(v1[1])
			v2[1] = self._flip_y(v2[1])
			v3[1] = self._flip_y(v3[1])
			v4[1] = self._flip_y(v4[1])
			v5[1] = self._flip_y(v5[1])
			pygame.draw.polygon(self.screen, col, [v1, v2, v3, v4, v5])
		elif shape is Shape.FAN:
			w, h = self.screen.get_width(), self.screen.get_height()
			max_len = math.sqrt(w * w + h * h)
			vl = transform.forward.rotated(0.5 * fan_angle) * max_len # TODO: length configurable?
			vr = transform.forward.rotated(-0.5 * fan_angle) * max_len # TODO: length configurable?
			end = (transform.position + vl).to_int_array()
			end[1] = self._flip_y(end[1])
			pygame.draw.aaline(self.screen, col, center, end, False)
			end = (transform.position + vr).to_int_array()
			end[1] = self._flip_y(end[1])
			pygame.draw.aaline(self.screen, col, center, end, False)
		else:
			raise NotImplementedError

	def update(self, world):
		self.screen.fill(Color["WHITE"])
		for o in world.objects:
			self._draw(o.shape, o.transform, o.color)
		for a in world.agents:
			self._draw(a.shape, a.transform, a.color)
			for s in a._sensors:
				self._draw(s.shape, s.transform, s.color, s.fov)
			#for s in a._wheels:
			#	self._draw(s.shape, s.transform, s.color)
		pygame.display.flip()


WORLD_SIZE = (500, 500)

def main():
	HRR.valid_range = [0,1]

	pygame.init()

	world = World(size=WORLD_SIZE)
	vis = Visualization(world.size)

	agent = Agent(color="RED", velocity=0.05)
	agent.transform.local_position = Vec2(10, 10)
	#agent.transform.local_orientation = Rot2(-math.pi / 4.0)
	agent.transform.local_orientation = Rot2(-math.pi / 3.0)
	sensor1 = Sensor(color="GREEN", field_of_view=0.2*math.pi)
	sensor1.transform.local_position = Vec2(-3, 0)
	sensor1.transform.local_orientation = Rot2(0.08 * math.pi)
	sensor2 = Sensor(color="BLUE", field_of_view=0.2*math.pi)
	sensor2.transform.local_position = Vec2(3, 0)
	sensor2.transform.local_orientation = Rot2(-0.08 * math.pi)
	agent.add_sensor(sensor1)
	agent.add_sensor(sensor2)
	world.add_agent(agent)

	obj1 = Object(color="CYAN")
	obj1.transform.local_position = Vec2(50, 50)
	obj2 = Object(color="MAGENTA")
	obj2.transform.local_position = Vec2(400, 70)
	obj3 = Object(color="YELLOW")
	obj3.transform.local_position = Vec2(80, 80)
	world.add_object(obj1)
	world.add_object(obj2)
	world.add_object(obj3)

	# create controller "follow magenta object"
	# direction gaussians should overlap to always yield a result
	HRR.stddev = 0.1
	# sensors return object in [0,1], 0 is left, 1 is right
	farleft = HRR(0.15)
	left = HRR(0.35)
	right = HRR(0.75)
	farright = HRR(0.85)
	# reset stddev to default
	HRR.stddev = 0.02
	# TODO: we don't have rule for driving forward?!
	farleft_ctl = farleft * (agent.wheels.left.hrr * HRR(WheelDir.BACKWARDS) + agent.wheels.right.hrr * HRR(WheelDir.FORWARDS))
	left_ctl = left * (agent.wheels.left.hrr * HRR(WheelDir.NONE) + agent.wheels.right.hrr * HRR(WheelDir.FORWARDS))
	right_ctl = right * (agent.wheels.left.hrr * HRR(WheelDir.FORWARDS) + agent.wheels.right.hrr * HRR(WheelDir.NONE))
	farright_ctl = farright * (agent.wheels.left.hrr * HRR(WheelDir.FORWARDS) + agent.wheels.right.hrr * HRR(WheelDir.BACKWARDS))
	follow_obj2_ctl = obj2.hrr * (farleft_ctl + left_ctl + right_ctl + farright_ctl)
	controller = obj1.hrr + follow_obj2_ctl + obj3.hrr
	agent.controller = controller

	clock = pygame.time.Clock()

	exit = False
	while(not exit):
		clock.tick(10) # param is FPS

		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				exit = True
				break

		#agent.transform.rotate(Rot2(0.001))
		#agent.transform.translate(Default.FORWARD)

		world.step(delta_time=1.0)
		vis.update(world)


def test_transform():
	#t0 = Transform()
	#t1 = Transform(parent=t0)
	#t1.local_position = Vec2(5.0, 3.0)
	#t1.orientation = Rot2(1.0 / 4.0 * 2.0 * math.pi)  # rotate 45 deg ccw
	#t2 = Transform(parent=t1)
	#t2.local_position = Vec2(1.0, 1.0)
	#t2.orientation = Rot2(-1.0 / 8.0 * 2.0 * math.pi)  # rotate 22.5 deg cw

	t0 = Transform()

	t1 = Transform(parent=t0)
	t1.local_position = Vec2(11.0, 5.0)
	t1.local_orientation = Rot2(0.5 * math.pi)  # rotate 45 deg ccw

	t2 = Transform(parent=t1)
	t2.local_position = Vec2(3.0, 3.0)
	t2.local_orientation = Rot2(-1.0/4.0 * math.pi)  # rotate 22.5 deg cw

	t3 = Transform(parent=t2)

	t4 = Transform(parent=t3)
	t4.local_position = Vec2(-5.65, 1.4)
	t4.local_orientation = Rot2(-1.0/4.0 * math.pi)  # rotate 22.5 deg cw

	print("t0: {}".format(t0))
	print("t1: {}".format(t1))
	print("t2: {}".format(t2))
	print("t3: {}".format(t3))
	print("t4: {}".format(t4))
	print("")
	print("t0.right: {}".format(t0.right))
	print("t0.forward: {}".format(t0.forward))
	print("t0.position: {}".format(t0.position))
	print("t1.right: {}".format(t1.right))
	print("t1.forward: {}".format(t1.forward))
	print("t1.position: {}".format(t1.position))
	print("t2.right: {}".format(t2.right))
	print("t2.forward: {}".format(t2.forward))
	print("t2.position: {}".format(t2.position))
	print("t3.right: {}".format(t3.right))
	print("t3.forward: {}".format(t3.forward))
	print("t3.position: {}".format(t3.position))
	print("t4.right: {}".format(t4.right))
	print("t4.forward: {}".format(t4.forward))
	print("t4.position: {}".format(t4.position))
	print("")
	print("t0  child in os: {}".format(t0.local_transformed_vector(t1.position)))
	print("t1 parent in os: {}".format(t1.local_transformed_vector(t0.position)))
	print("t1  child in os: {}".format(t1.local_transformed_vector(t2.position)))
	print("t2 parent in os: {}".format(t2.local_transformed_vector(t1.position)))
	print("t2  child in os: {}".format(t2.local_transformed_vector(t3.position)))
	print("t3 parent in os: {}".format(t3.local_transformed_vector(t2.position)))
	print("t3  child in os: {}".format(t3.local_transformed_vector(t4.position)))
	print("t4 parent in os: {}".format(t4.local_transformed_vector(t3.position)))


if __name__ == "__main__":
	#test_transform()
	main()

