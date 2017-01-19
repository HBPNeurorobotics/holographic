import math
import pygame
import random

import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from pygame.locals import *
from collections import deque, OrderedDict

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


Color = OrderedDict([
	("WHITE"  , (255, 255, 255)),
	("BLACK"  , (  0,   0,   0)),
	("RED"    , (255,   0,   0)),
	("GREEN"  , (  0, 255,   0)),
	("BLUE"   , (  0,   0, 255)),
	("CYAN"   , (  0, 255, 255)),
	("MAGENTA", (255,   0, 255)),
	("YELLOW" , (255, 255,   0)),
])

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
		self.velocity = velocity
		self._direction = self.direction = wheel_dir

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
	def direction(self):
		return self._direction

	@direction.setter
	def direction(self, value):
		assert isinstance(value, WheelDir)
		self._direction = value

	@property
	def value(self):
		# NOTE: hack from Igor (always move forwards)
		return self._dir_to_val(self._direction) * self.velocity + 0.2


class WheelSet(object):
	def __init__(self, velocity=1.0, left=WheelDir.NONE, right=WheelDir.NONE):
		self._left = Wheel(velocity, left)
		self._right = Wheel(velocity, right)

	@property
	def left(self):
		return self._left

	@left.setter
	def left(self, value):
		self._left.direction = value

	@property
	def right(self):
		return self._right

	@right.setter
	def right(self, value):
		self._right.direction = value


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
				output.append((o, pos_vs, rel_pos.length()))
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
					output = HRR(o) * pos_vs
				else:
					output += HRR(o) * pos_vs
		return HRR(self) * output if output is not None else HRR(self)


class Controller(object):
	def __init__(self, controller, sensor):
		self.NO_OBJECT = None
		self.controller = controller
		self.sensor = sensor


class Agent(VisObject):
	TARGET_NEW_TIME = 1

	def __init__(self, color, velocity=1.0):
		super(Agent, self).__init__(Shape.HOUSE, color)
		self._target = self.target = 0
		self.controllers = []
		self._sensors = []
		self.steering = DiffSteer()
		self.wheels = WheelSet(velocity=velocity)

		# state variables - read only
		self.target_position_vs = np.NAN
		self.target_distance = np.NAN
		self.similarity_left = [np.NAN, np.NAN]
		self.similarity_right = [np.NAN, np.NAN]
		self.target_new = 0

	def __str__(self):
		return super(Agent, self).__str__()

	@property
	def target(self):
		return self._target

	@target.setter
	def target(self, value):
		self.target_new = Agent.TARGET_NEW_TIME + 1
		self._target = value

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
		self.target_new = max(0, self.target_new - 1) # decrease "target new time"

		self.wheels.left = WheelDir.NONE
		self.wheels.right = WheelDir.NONE

		for c in self.controllers:
			output = c.sensor.read(objects)

			max_similarity_left = 0.1
			wheel_dir_left = None
			max_similarity_right = 0.1
			wheel_dir_right = None

			self.target_position_vs = np.NAN
			self.target_distance = np.NAN

			# TODO: make it more beautiful
			for obj, val, dist in output:
				if obj is self.target:
					self.target_position_vs = val
					self.target_distance = dist

					actions_ctl = c.controller % val
					actions = actions_ctl / HRR(self.wheels.left)
					print("val: {} l: {}".format(val, actions))
					for a in actions:
						wd = None
						try: wd = WheelDir(a[0])
						except: pass
						if wd is not None and a[1] > max_similarity_left:
							wheel_dir_left = wd
							max_similarity_left = a[1]
						if wd is WheelDir.FORWARDS:
							self.similarity_left[0] = a[1]
						elif wd is WheelDir.BACKWARDS:
							self.similarity_left[1] = a[1]
					actions = actions_ctl / HRR(self.wheels.right)
					print("val: {} r: {}".format(val, actions))
					for a in actions:
						wd = None
						try: wd = WheelDir(a[0])
						except: pass
						if wd is not None and a[1] > max_similarity_right:
							wheel_dir_right = wd
							max_similarity_right = a[1]
						if wd is WheelDir.FORWARDS:
							self.similarity_right[0] = a[1]
						elif wd is WheelDir.BACKWARDS:
							self.similarity_right[1] = a[1]

			if wheel_dir_left is None or wheel_dir_right is None:
				print("NO WHEEL ACTION FOUND! l {} r {} tp {}".format(wheel_dir_left, wheel_dir_right, self.target_position_vs))

				self.similarity_left = [np.NAN, np.NAN]
				self.similarity_right = [np.NAN, np.NAN]

				# we don't have results for both wheels - probe for NO_OBJECT
				actions_ctl = c.controller % c.NO_OBJECT
				actions = actions_ctl / HRR(self.wheels.left)
				for a in actions:
					wd = None
					try: wd = WheelDir(a[0])
					except: pass
					if wd is not None and a[1] > max_similarity_left:
						wheel_dir_left = wd
						max_similarity_left = a[1]
					if wd is WheelDir.FORWARDS:
						self.similarity_left[0] = a[1]
					elif wd is WheelDir.BACKWARDS:
						self.similarity_left[1] = a[1]
				actions = actions_ctl / HRR(self.wheels.right)
				for a in actions:
					wd = None
					try: wd = WheelDir(a[0])
					except: pass
					if wd is not None and a[1] > max_similarity_right:
						wheel_dir_right = wd
						max_similarity_right = a[1]
					if wd is WheelDir.FORWARDS:
						self.similarity_right[0] = a[1]
					elif wd is WheelDir.BACKWARDS:
						self.similarity_right[1] = a[1]

			self.wheels.left = wheel_dir_left
			self.wheels.right = wheel_dir_right

		print ("al: {} ar: {}".format(self.wheels.left.direction , self.wheels.right.direction))

		pos, rot = self.steering.move(self.wheels,
				self.transform.local_position,
				self.transform.local_orientation, delta_time)

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
		return obj

	def add_random_object(self):
		idx = int(1 + ((len(Color.keys()) - 1) * random.random()))
		obj = Object(color=Color.keys()[idx])
		x = random.random() * (self.size[0] - 1)
		y = random.random() * (self.size[1] - 1)
		obj.transform.local_position = Vec2(x, y)
		return self.add_object(obj)

	def remove_target_object(self, agent):
		self.objects.remove(agent.target)
		agent.target = None

	def step(self, delta_time):
		for a in self.agents:
			a.step(self.objects, delta_time)
			# check world space limits
			a.transform.local_position.x = min(max(a.transform.local_position.x, 0.0), self.size[0])
			a.transform.local_position.y = min(max(a.transform.local_position.y, 0.0), self.size[1])


class Visualization(object):
	def __init__(self, size, plot_pipeline=True, num_pipeline_entries=1000):
		self.screen = pygame.display.set_mode(size)
		self.plot_pipeline = plot_pipeline

		d1 = np.full(num_pipeline_entries, np.NAN, dtype='float32')
		d2 = np.full(num_pipeline_entries, [np.NAN, np.NAN], dtype='2float32')
		self._pipeline_inputs = deque(d1, num_pipeline_entries)
		self._pipeline_similarity_l = deque(d2, num_pipeline_entries)
		self._pipeline_similarity_r = deque(d2, num_pipeline_entries)
		self._pipeline_distance = deque(d1, num_pipeline_entries)
		self._pipleline_new_target = deque(d1, num_pipeline_entries)

		pygame.display.set_caption("robosim")
		self.screen.fill(Color["WHITE"])
		pygame.display.flip()

		plt.ion()

		self.fig = plt.figure(figsize=(6,10))
		self.ax1 = self.fig.add_subplot(4, 1, 1)
		self.ax2 = self.fig.add_subplot(4, 1, 2)
		self.ax3 = self.fig.add_subplot(4, 1, 3)
		self.ax4 = self.fig.add_subplot(4, 1, 4)

		#self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
		self.ax1.set_xlim(0, num_pipeline_entries)
		self.ax1.set_xticks([])
		self.ax1.set_xlabel("Sensory Input")
		self.ax1.set_ylim(0, 1)
		#self.ax1.set_ylim(self.ax1.get_ylim()[::-1])
		#self.ax1.set_yticks([])
		sens_series = np.array(self._pipeline_inputs).astype(np.double)
		sens_mask1 = np.isfinite(sens_series)
		sens_mask2 = np.isnan(sens_series)
		self.line11, = self.ax1.plot(
				np.arange(num_pipeline_entries)[sens_mask1],
				sens_series[sens_mask1],
				label="Similarity",
				lw=1.5,
				alpha=0.7,
				c=[1.0, 0.31, 0.0])
		self.line12, = self.ax1.plot(
				np.arange(num_pipeline_entries)[sens_mask2],
				sens_series[sens_mask2],
				label="Similarity",
				lw=1.5,
				alpha=0.9,
				c=[1.0, 0.0, 0.0])

		self.ax2.set_xlim(0, num_pipeline_entries)
		self.ax2.set_xticks([])
		self.ax2.set_xlabel("Similarity Left Wheel")
		self.ax2.set_ylim(-0.4, 0.4)
		self.line21, = self.ax2.plot(
				np.arange(num_pipeline_entries),
				[a for a,_ in self._pipeline_similarity_l],
				label="Forwards",
				lw=1.5,
				alpha=0.7,
				c=[1.0, 0.31, 0.0])
		self.line22, = self.ax2.plot(
				np.arange(num_pipeline_entries),
				[b for _,b in self._pipeline_similarity_l],
				label="Backwards",
				lw=1.5,
				alpha=0.7,
				c=[0.0, 0.31, 1.0])
		self.line23, = self.ax2.plot(
				np.arange(num_pipeline_entries),
				[a for a,_ in self._pipeline_similarity_r],
				label="Forwards",
				lw=1.5,
				alpha=0.7,
				c=[1.0, 0.31, 0.0])
		self.line24, = self.ax2.plot(
				np.arange(num_pipeline_entries),
				[b for _,b in self._pipeline_similarity_r],
				label="Backwards",
				lw=1.5,
				alpha=0.7,
				c=[0.0, 0.31, 1.0])
		self.ax2.legend(loc="lower left", handles=[self.line21, self.line22, self.line23, self.line24])

		self.ax3.set_xlim(0, num_pipeline_entries)
		self.ax3.set_xticks([])
		self.ax3.set_xlabel("Similarity Right Wheel")
		self.ax3.set_ylim(-0.4, 0.4)
		self.line31, = self.ax3.plot(
				np.arange(num_pipeline_entries),
				[a for a,_ in self._pipeline_similarity_r],
				label="Forwards",
				lw=1.5,
				alpha=0.7,
				c=[1.0, 0.31, 0.0])
		self.line32, = self.ax3.plot(
				np.arange(num_pipeline_entries),
				[b for _,b in self._pipeline_similarity_r],
				label="Backwards",
				lw=1.5,
				alpha=0.7,
				c=[0.0, 0.31, 1.0])
		self.ax3.legend(loc="lower left", handles=[self.line31, self.line32])

		self.ax4.set_xlim(0, num_pipeline_entries)
		self.ax4.set_xticks([])
		self.ax4.set_xlabel("Distance to Target")
		#w, h = self.screen.get_width(), self.screen.get_height()
		#max_len = math.sqrt(w * w + h * h)
		max_len = 500
		self.ax4.set_ylim(0, max_len)
		self.line4, = self.ax4.plot(
				np.arange(num_pipeline_entries),
				self._pipeline_distance,
				label="Distance",
				lw=1.5,
				alpha=0.7,
				c=[1.0, 0.31, 0.0])

		if self.plot_pipeline:
			plt.show()

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
			l = 7.0
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

	def _update_symbolic_pipeline(self, agent):
		self._pipeline_inputs.append(agent.target_position_vs)
		self._pipeline_similarity_l.append(agent.similarity_left)
		self._pipeline_similarity_r.append(agent.similarity_right)
		self._pipeline_distance.append(Vec2.length(agent.transform.position - agent.target.transform.position))
		self._pipleline_new_target.append(agent.target_new > 0)

	def _update_symbolic_pipeline_plot(self):
		sens_series = np.array(self._pipeline_inputs).astype(np.double)
		sens_mask1 = np.isfinite(sens_series)
		sens_mask2 = np.isnan(sens_series)
		self.line11.set_data(np.arange(len(self._pipeline_inputs))[sens_mask1], sens_series[sens_mask1])
		self.line12.set_data(np.arange(len(self._pipeline_inputs))[sens_mask2], sens_series[sens_mask2])

		self.line21.set_data(np.arange(len(self._pipeline_similarity_l)), [a for a,_ in self._pipeline_similarity_l])
		self.line22.set_data(np.arange(len(self._pipeline_similarity_l)), [b for _,b in self._pipeline_similarity_l])
		self.line23.set_data(np.arange(len(self._pipeline_similarity_r)), [a for a,_ in self._pipeline_similarity_r])
		self.line24.set_data(np.arange(len(self._pipeline_similarity_r)), [b for _,b in self._pipeline_similarity_r])
		self.line31.set_data(np.arange(len(self._pipeline_similarity_r)), [a for a,_ in self._pipeline_similarity_r])
		self.line32.set_data(np.arange(len(self._pipeline_similarity_r)), [b for _,b in self._pipeline_similarity_r])
		#self.line4.set_data(np.arange(len(self._pipeline_distance)), self._pipeline_distance)

		# yeah, matplotlib is so retarded...
		self.ax4.cla()
		self.ax4.set_xlim(0, len(self._pipleline_new_target))
		self.ax4.set_xticks([])
		self.ax4.set_xlabel("Distance to Target")
		#w, h = self.screen.get_width(), self.screen.get_height()
		#max_len = math.sqrt(w * w + h * h)
		max_len = 500
		self.ax4.set_ylim(0, max_len)
		for x, v in enumerate(self._pipleline_new_target):
			if v is True:
				self.ax4.axvline(x=x, color=[0.0, 0.0, 0.0],
						lw=1.0, alpha=0.8)
		self.line4, = self.ax4.plot(
				np.arange(len(self._pipeline_distance)),
				self._pipeline_distance,
				label="Distance",
				lw=1.5,
				alpha=0.7,
				c=[1.0, 0.31, 0.0])

		self.fig.canvas.draw()

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

		if (self.plot_pipeline and len(world.agents) >= 1):
			# only plot for the first agent
			self._update_symbolic_pipeline(world.agents[0])
			self._update_symbolic_pipeline_plot()


WORLD_SIZE = (500, 500)

def main():
	HRR.valid_range = zip([0.0], [1.0])
	HRR.set_size(4096)

	pygame.init()

	world = World(size=WORLD_SIZE)
	vis = Visualization(world.size)

	agent = Agent(color="RED", velocity=0.07)
	agent.transform.local_position = Vec2(10, 10)
	#agent.transform.local_orientation = Rot2(-math.pi / 3.0)
	agent.transform.local_orientation = Rot2(random.random() * 2.0 * math.pi)
	sensor1 = Sensor(color="GREEN", field_of_view=0.2*math.pi)
	#sensor1.transform.local_position = Vec2(3, 0)
	#sensor1.transform.local_orientation = Rot2(0.08 * math.pi)
	agent.add_sensor(sensor1)
	world.add_agent(agent)

	#obj1 = Object(color="CYAN")
	#obj1.transform.local_position = Vec2(50, 50)
	#obj2 = Object(color="MAGENTA")
	#obj2.transform.local_position = Vec2(400, 70)
	#obj3 = Object(color="YELLOW")
	#obj3.transform.local_position = Vec2(80, 80)
	#world.add_object(obj1)
	#world.add_object(obj2)
	#world.add_object(obj3)
	world.add_random_object()
	world.add_random_object()
	target_obj = world.add_random_object()

	# create controller "follow magenta object"
	# direction gaussians should overlap to always yield a result
	HRR.stddev = 0.05

	# sensors return object in [0,1], 0 is left, 1 is right
	farleft = HRR(0.1)
	left = HRR(0.3)
	front = HRR(0.5)
	right = HRR(0.7)
	farright = HRR(0.9)

	# test stddev and result
	#farleft.plot(unpermute=True)
	#left.plot(unpermute=True)
	#forwards.plot(unpermute=True)
	#right.plot(unpermute=True)
	#farright.plot(unpermute=True)
	#temp = farleft + left + forwards + right + farright
	#temp.plot(unpermute=True)

	# reset stddev to default
	HRR.stddev = 0.02

	left_wheel_backwards = HRR(agent.wheels.left) * HRR(WheelDir.BACKWARDS)
	left_wheel_forwards = HRR(agent.wheels.left) * HRR(WheelDir.FORWARDS)
	right_wheel_backwards = HRR(agent.wheels.right) * HRR(WheelDir.BACKWARDS)
	right_wheel_forwards = HRR(agent.wheels.right) * HRR(WheelDir.FORWARDS)

	farleft_ctl = farleft * (left_wheel_backwards + right_wheel_forwards)
	left_ctl = left * (left_wheel_backwards + right_wheel_forwards)
	front_ctl = front * (left_wheel_forwards + right_wheel_forwards)
	right_ctl = right * (left_wheel_forwards + right_wheel_backwards)
	farright_ctl = farright * (left_wheel_forwards + right_wheel_backwards)

	NO_OBJECT = HRR("NO_OBJECT")
	no_object_ctl = NO_OBJECT * (left_wheel_forwards + right_wheel_backwards)

	sensor_ctl = farleft_ctl + left_ctl + front_ctl + right_ctl + farright_ctl + no_object_ctl

	print(sensor_ctl.distance((sensor_ctl % farleft).memory, (left_wheel_backwards + right_wheel_forwards).memory))

	agent.controllers.append(Controller(sensor_ctl, sensor1))
	agent.controllers[0].NO_OBJECT = NO_OBJECT
	agent.target = target_obj

	clock = pygame.time.Clock()

	exit = False
	while(not exit):
		clock.tick(10) # param is FPS

		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				exit = True
				break

		world.step(delta_time=1.0)
		vis.update(world)

		# remove and add new target if agent is close
		if agent.target_distance < 50:
			world.remove_target_object(agent)
			target_obj = world.add_random_object()
			agent.target = target_obj


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

