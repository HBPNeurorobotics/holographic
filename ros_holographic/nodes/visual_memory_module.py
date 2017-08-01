#!/usr/bin/env python

import rospy
import sys
from ros_holographic.srv import * 
from vsa.hrr import HRR
import numpy as np

class HRR_Node: 

  m = None

  def handle_clear_memory(self, req):
      self.m = None
      return ClearMemoryResponse(True)
      
  def handle_new_object(self, req):
      if not HRR.valid_range[0][0] < req.X < HRR.valid_range[0][1] or not HRR.valid_range[1][0] < req.X < HRR.valid_range[1][1]:
	print "Coordinates outside valid range"
	return NewObjectResponse(False)
      if self.m is None:
	self.m = HRR(req.Label) * (req.X, req.Y)
      else:
	self.m += HRR(req.Label) * (req.X, req.Y)
      return NewObjectResponse(True)

  def handle_coordinate_probe(self, req):
      if not HRR.valid_range[0][0] < req.X < HRR.valid_range[0][1] or not HRR.valid_range[1][0] < req.X < HRR.valid_range[1][1]:
	print "Coordinates outside valid range"
	return ProbeCoordinateResponse(False, None)
      elif self.m is None:
	print "Memory is empty"
	return ProbeCoordinateResponse(False, None)
      d = self.m / (req.X, req.Y)
      l = []
      for s, v in d:
	if v >= HRR.distance_threshold:
	  l.append(s)
      return ProbeCoordinateResponse(True, l)

  def handle_label_probe(self, req):
      if self.m is None:
	print "Memory is empty"
	return ProbeLabelResponse(False, None)
      out = self.m % req.Label
      d = out.decodeCoordinate(dim=2, return_list=True)
      l1 = []
      l2 = []
      for s, v in d:
	if v == 1:
	  l1.append(s[0])
	  l2.append(s[1])
      return ProbeLabelResponse(True, l1, l2)
    
  def visual_scene_memory_server(self):
      rospy.init_node('visual_scene_memory_server')
      s0 = rospy.Service('clear_memory', ClearMemory, self.handle_clear_memory)
      s1 = rospy.Service('new_object', NewObject, self.handle_new_object)
      s2 = rospy.Service('probe_coordinate', ProbeCoordinate, self.handle_coordinate_probe)
      s3 = rospy.Service('probe_label', ProbeLabel, self.handle_label_probe)
      print "HRR Memory Node is ready..."
      rospy.spin()

if __name__ == "__main__":
  HRR.peak_min = 0.1 # May need to be adjusted for label probe. If fewer coordinates are returned than expected lower value. If new coordinates that don't belong are returnd increase value.
  HRR.stddev = 0.02 # Percentual area in which a label will be returned from a coordinate. I.e. probing for (19,19) and receiving a label at (20,20). Lower if this effect is not desired.
  HRR.distance_threshold = 0.2 # Minimum similarity a label must present to be accepted as a result. Increasing this will make probe coordinate more strict.
  HRR.return_list = True
  HRR.size = 10000
  HRR.valid_range = ((-100,100),(-100,100))
  HRR_Node().visual_scene_memory_server()