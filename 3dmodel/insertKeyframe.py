"""
insert the prediction parameters into blender file model.blend and visualize the
prediction in order to compare with the original ones.
"""
from bpy import data, ops, props, types, context
from math import degrees
import json
import bpy, bgl, blf,sys
import math

paras = json.load(open('../results/predictions_cnn_25.json'))
frames = list(paras.keys())
frames.sort(key=lambda name: int(name.strip().replace('"',"")))
# choose which object to record
obj = bpy.data.objects['boat1_prediction']
scene = bpy.context.scene
for key in frames:
    key = str(key)
    value = paras[key]
    key = int(key)
    pi = math.pi
    # set current frame
    scene.frame_set(int(key))
    # rad -> degree
    obj.rotation_euler = (value[0]*pi/180, value[1]*pi/180, 90*pi/180)
    # insert roll(index=0) and pitch(index=1)
    obj.keyframe_insert(data_path='rotation_euler', index=0, frame=key)
    obj.keyframe_insert(data_path='rotation_euler', index=1, frame=key)
    print("insert keyframe:", key)
print("Done!")
