import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json
# setup gpu render
bpy.context.user_preferences.system.compute_device_type = 'CUDA'
bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_2'

print('\nPrint Scenes...')
paras = {}
scene = bpy.context.scene
for i in range(1,5000):
    scene.frame_set(i)
    scene.render.filepath = './render619/' + str(i)
    bpy.ops.render.render(write_still=True)
    # get paras
    loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
    rot = rot.to_euler()
    rot = list(degrees(a) for a in rot)
    paras.update({i:rot})
jsObj = json.dumps(paras)
fileObject = open('./render619/paras_origin.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print(paras)
print('Done!')
