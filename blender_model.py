import bpy
from math import degrees

bpy.data.objects['boat'].matrix_world.decompose()

loc, rot, scale = obj.matrix_world.decompose()
bpy.data.objects['boat'].rotation_euler[1]
loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
b = rot.to_euler()
print("%.2f, %.2f, %.2f" % tuple(degrees(a) for a in b))
#v1 = [1, 1, 1]
#v2 = [0, 1, 1]
#print(degrees(v1.angle(v2)))
# cam = bpy.data.objects[‘Camera.1’]
# posX = cam.location.x
# posY = cam.location.y
# posZ = cam.location.z
cam = bpy.data.objects[‘Camera’]
rotX = cam.rotation_euler.x
rotY = cam.rotation_euler.y
rotZ = cam.rotation_euler.z

scene = bpy.context.scene
fp = scene.render.filepath # get existing output path
scene.render.image_settings.file_format = 'PNG' # set output format to .png

frames = 50, 90, 170

for frame_nr in frames:

    # set current frame to frame 5
    scene.frame_set(frame_nr)

    # set output path so render won't get overwritten
    scene.render.filepath = fp + str(frame_nr)
    bpy.ops.render.render(write_still=True) # render still

# restore the filepath
scene.render.filepath = fp


"""
# below works
"""
# Render all cameras or cameras containing text given with command line argument "cameras".
# Example:
# Let's say test.blend file contains cameras "east.01", "east.02", "west.01", "west.02"
# By executing command "blender -b your_file.blend -P render_all_cameras.py" all 4 cameras will be rendered.
# Command "blender -b your_file.blend -P render_all_cameras.py  cameras=east" will render "east.01" and "east.02" cameras.
# Command "blender -b your_file.blend -P render_all_cameras.py  cameras=01" will render "east.01" and "west.01.02" cameras.


import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
print("\nThis Python script will render your scene with all available cameras")
print("or with camera(s) matching command line argument 'cameras'")
print("")
print("Usage:")
print("Render all cameras:")
print("blender -b your_file.blend -P render_all_cameras.py\n")
print("Render only matching cameras:")
print("blender -b your_file.blend -P render_all_cameras.py  cameras=east\n")

cameraNames=''

# Loop all command line arguments and try to find "cameras=east" or similar
for arg in sys.argv:
    words=arg.split('=')
    if ( words[0] == 'cameras'):
     cameraNames = words[1]

print('rendering cameras containing [' + cameraNames + ']')

print('\nPrint Scenes...')
sceneKey = bpy.data.scenes.keys()[0]
print('Using Scene['  + sceneKey + ']')

# Loop all objects and try to find Cameras
print('Looping Cameras')
c=0
for obj in bpy.data.objects:
    # Find cameras that match cameraNames
    if ( obj.type =='CAMERA') and ( cameraNames == '' or obj.name.find(cameraNames) != -1) :
      print("Rendering scene["+sceneKey+"] with Camera["+obj.name+"]")

      # Set Scenes camera and output filename
      bpy.data.scenes[sceneKey].camera = obj
      #bpy.data.scenes[sceneKey].render.file_format = 'JPEG'
      bpy.data.scenes[sceneKey].render.filepath = '//camera_' + str(c) + '_'

      # Render Scene and store the scene
      bpy.ops.render.render( animation=True )
      c = c + 1
print('Done!')

"""
or below
"""
import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json#, GameLogic

cameraNames=''
for arg in sys.argv:
    words=arg.split('=')
    if ( words[0] == 'cameras'):
     cameraNames = words[1]

print('\nPrint Scenes...')
sceneKey = bpy.data.scenes.keys()[0]
print('Using Scene['  + sceneKey + ']')
paras = {}
scene = bpy.context.scene
for i in range(1000):
    scene.frame_set(i*10)
    scene.render.filepath = '/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/img/' + str(i*10)
    bpy.ops.render.render(write_still=True)

      # get paras
    loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
    # r = bpy.data.objects['boat'].rotation_euler
    rot = rot.to_euler()
    rot = list(degrees(a) for a in rot)
    paras.update({i*10:rot})
    #c = c + 1
print(paras)
jsObj = json.dumps(paras)

fileObject = open('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/paras.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print('Done!')

"""
try get angle
"""
matrix_rotation = owner.worldOrientation
euler_rotation = matrix_rotation.to_euler()
degrees_rotation = [math.degrees(a) for a in euler_rotation]

degrees = int(degrees_rotation[1])

print(degrees)matrix_rotation = owner.worldOrientation
euler_rotation = matrix_rotation.to_euler()
degrees_rotation = [math.degrees(a) for a in euler_rotation]

degrees = int(degrees_rotation[1])

print(degrees)

def get_rotation(object):
    matrix_rotation = object.worldOrientation
    euler_rotation = matrix_rotation.to_euler()
    degrees_rotation = [math.degrees(a) for a in euler_rotation]
    print(degrees_rotation)
    return degrees_rotation

# get current controller
controller = GL.getCurrentController()

# get object script attached to
obj = controller.owner
get_rotation(obj)


"""
this is right
"""
from bpy import data, ops, props, types, context
from math import degrees
import json#, GameLogic
import bpy, bgl, blf,sys

paras = json.load(open('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/paras.json'))
frames = list(paras.keys())
frames.sort(key=lambda name: int(name.strip().replace('"',"")))
obj = bpy.data.objects['boat2']
scene = bpy.context.scene
#for key in range(10):
for key in frames:
    key = str(key)
    value = paras[key]
    #value = [0.0081*int(key), -0.0052*int(key), 0.073*int(key)]
#for key, value in paras.items():
#    key = key.replace('"',"")
    key = int(key)
    #print(type(value))

    #print(type(value[2]))
    if key<=1000 :
        scene.frame_set(int(key))
        #loc = obj1.location
        #print(loc)
        #obj.location = (5, 5, key*2)
        #obj.location = (0.5954, -8.2680, 0.6732)
        #obj.rotation_euler = Euler((value[0], value[1], value[2]), 'XYZ')
        print(key, value[0], value[1], value[2])
        obj.rotation_euler = (value[0]*3.14159/180, value[1]*3.14159/180, value[2]*3.14159/180)
        #obj.keyframe_insert(data_path='location', index=2, frame=key)
       # obj.keyframe_delete('rotation_euler', frame=key)
        #obj.keyframe_delete('location', frame=key)

        obj.keyframe_insert(data_path='rotation_euler', index=0, frame=key)
        obj.keyframe_insert(data_path='rotation_euler', index=1, frame=key)
"""
this works
"""
import bpy
from math import degrees
import math
scene = bpy.context.scene
scene.frame_set(18)
#if 'Z' in bpy.data.objects['boat'].rotation_mode:
r = bpy.data.objects['boat'].rotation_euler
#print(math.degrees(r[0]))
print(len(r))
print("%.2f, %.2f, %.2f" % tuple(degrees(a) for a in r))
#for i in r:
#    print(i)

print("---------")
loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
b = rot.to_euler()
print("%.2f, %.2f, %.2f" % tuple(degrees(a) for a in b))

"""
record parameters
"""
import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json

cameraNames=''
for arg in sys.argv:
    words=arg.split('=')
    if ( words[0] == 'cameras'):
     cameraNames = words[1]

print('\nPrint Scenes...')
sceneKey = bpy.data.scenes.keys()[0]
print('Using Scene['  + sceneKey + ']')
paras = {}
scene = bpy.context.scene
for i in range(1000):
    scene.frame_set(i*10)
    scene.render.filepath = '/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/img/' + str(i)
    bpy.ops.render.render(write_still=True)

      # get paras
    loc, rot, scale = bpy.data.objects['boat'].matrix_world.decompose()
    # r = bpy.data.objects['boat'].rotation_euler
    rot = rot.to_euler()
    rot = list(degrees(a) for a in rot)
    paras.update({i*10:rot})
print(paras)
jsObj = json.dumps(paras)
fileObject = open('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/paras.json', 'w')
fileObject.write(jsObj)
fileObject.close()
print('Done!')


"""
insert keyframe
"""
from bpy import data, ops, props, types, context
from math import degrees
import json
import bpy, bgl, blf,sys
import math

paras = json.load(open('/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/3dmodel/test.json'))
frames = list(paras.keys())
frames.sort(key=lambda name: int(name.strip().replace('"',"")))
obj = bpy.data.objects['boat2']
scene = bpy.context.scene
for key in frames:
    key = str(key)
    value = paras[key]
    key = int(key)
    pi = math.pi
    scene.frame_set(int(key))
    #obj.rotation_euler = Euler((value[0], value[1], value[2]), 'XYZ')
    print(key, value[0], value[1], value[2])
    obj.rotation_euler = (value[0]*pi/180, value[1]*pi/180, value[2]*pi/180)
    #obj.keyframe_insert(data_path='location', index=2, frame=key)
    #obj.keyframe_delete('rotation_euler', frame=key)
    #obj.keyframe_delete('location', frame=key)

    obj.keyframe_insert(data_path='rotation_euler', index=0, frame=key)
    obj.keyframe_insert(data_path='rotation_euler', index=1, frame=key)
    #obj.keyframe_insert(data_path='rotation_euler', index=2, frame=key)
"""
delete keyframes
"""
from bpy import data, ops, props, types, context
from math import degrees
import json
import bpy, bgl, blf,sys
import math

obj = bpy.data.objects['boat2']
scene = bpy.context.scene
for key in range(0,5000,5):

    scene.frame_set(int(key))
    #obj.keyframe_insert(data_path='location', index=2, frame=key)
    obj.keyframe_delete('rotation_euler', frame=key)
    #obj.keyframe_delete('location', frame=key)
