"""
File containing all the constants used in the different files
"""
import numpy as np

# Main Constants
CAMERA_RESOLUTION = (486, 270)
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
MAX_HEIGHT = CAMERA_RESOLUTION[1]

ROI = [0, 0, MAX_WIDTH, MAX_HEIGHT]
# predict how many frames ahead
TIME_GAP = 25
TOTAL_FRAME = 5000
# dataset sample frame interval
FRAME_INTERVAL = 1
# concatenate two images as input?
ONE_IMG_ONLY = True
# if yes, what is the frame gap between two images
TWO_IMG_GAP = 3
# Training
FACTOR = 3  # Resize factor
INPUT_HEIGHT = ROI[3] // FACTOR
INPUT_WIDTH = ROI[2] // FACTOR
SPLIT_SEED = 42  # For train/val/test split
WEIGHTS_PTH = "cnn_model.pth"  # Path to the trained model
RES_DIR = "./Pre/results/"

# Arrow keys
# modified by zhi
UP_KEY = 82 # 0
DOWN_KEY = 84 # 1
RIGHT_KEY = 83 #3
LEFT_KEY = 81 # 2
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # S key
