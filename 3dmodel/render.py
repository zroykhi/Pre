import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json

print('\nPrint Scenes...')
paras = {}
scene = bpy.context.scene
for i in range(10):
    scene.frame_set(i*10)
    scene.render.filepath = './render/' + str(i)
    bpy.ops.render.render(write_still=True)
    # get paras
    loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
    rot = rot.to_euler()
    rot = list(degrees(a) for a in rot)
    paras.update({i*10:rot})
jsObj = json.dumps(paras)
fileObject = open('./render/paras_origin.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print(paras)
print('Done!')
