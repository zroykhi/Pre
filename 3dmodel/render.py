"""
this code is used for render images data set with blender file
"""
import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json
from constants import FRAME_INTERVAL, TOTAL_FRAME
IMG_DIR="./render619/"
CUDA=True

# setup gpu render if your computer supports
# remmenber change blender render setting: tiles center:x:512, y:512 for gpu support
# otherwise tiles center:x:16, y:16 for rendering with cpu only
if CUDA:
    bpy.context.user_preferences.system.compute_device_type = 'CUDA'
    bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_2'
    for scene in bpy.data.scenes:
    scene.render.tile_x = 512
    scene.render.tile_y = 512
    scene.cycles.tile_order = 'CENTER'
    
print('\nPrint Scenes...')
paras = {}
scene = bpy.context.scene
# RUN small test before running the real one
# for i in range(10):
for i in range(1,int(TOTAL_FRAME/FRAME_INTERVAL)+1):
    scene.frame_set(i*FRAME_INTERVAL)
    scene.render.filepath = IMG_DIR + str(i*FRAME_INTERVAL)
    bpy.ops.render.render(write_still=True)
    # get paras
    loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
    rot = rot.to_euler()
    rot = list(degrees(a) for a in rot)
    paras.update({i*FRAME_INTERVAL:rot})
jsObj = json.dumps(paras)
fileObject = open(IMG_DIR+'paras_origin.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print(paras)
print('Done!')
