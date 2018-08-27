"""
this code is used for render images data set with blender file
"""
import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json
import sys
import os
dirpath = os.getcwd()
sys.path.append(dirpath)
FRAME_INTERVAL = 1
START_FRAME = 1
END_FRAME = 5000
BOAT = 'boat'
CAMERA = 'Camera_fixtoboat_stable'
IMG_DIR="./render/"

"""
setup gpu render if your computer supports
remmenber change blender render setting: tiles center:x:512, y:512 for gpu support
otherwise tiles center:x:16, y:16 for rendering with cpu only
"""
# CUDA=False
# if CUDA:
#     bpy.context.user_preferences.system.compute_device_type = 'CUDA'
#     bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_2'
#     for scene in bpy.data.scenes:
#         scene.render.tile_x = 512
#         scene.render.tile_y = 512
#         scene.cycles.tile_order = 'CENTER'
# else:
#     for scene in bpy.data.scenes:
#         scene.render.tile_x = 16
#         scene.render.tile_y = 16
#         scene.cycles.tile_order = 'CENTER'
print('\nPrint Scenes...')
paras = {}
context = bpy.context
scene = context.scene
currentCameraObj = bpy.data.objects[CAMERA]
scene.camera = currentCameraObj
# RUN small test before running the real one
# for i in range(10):
for i in range(START_FRAME,int(END_FRAME/FRAME_INTERVAL)+1):
    print('current frame:', i)
    # set current frame
    scene.frame_set(i*FRAME_INTERVAL)
    scene.render.filepath = IMG_DIR + str(i*FRAME_INTERVAL)
    bpy.ops.render.render(write_still=True)
    # get paras (degree)
    loc, rot, scale = bpy.data.objects[BOAT].matrix_world.decompose()
    rot = rot.to_euler()
    rot = list(degrees(a) for a in rot)
    paras.update({i*FRAME_INTERVAL:rot})
jsObj = json.dumps(paras)
fileObject = open(IMG_DIR+'paras_origin.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print(paras)
print('Done!')
