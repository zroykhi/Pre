"""
File containing all the constants used in the different files
"""
import numpy as np

# Main Constants
# CAMERA_RESOLUTION = (800, 600)
CAMERA_RESOLUTION = (960, 540)
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
MAX_HEIGHT = CAMERA_RESOLUTION[1]
# r = [margin_left, margin_top, width, height]
ROI = [0, 0, MAX_WIDTH, MAX_HEIGHT]

# Training
FACTOR = 8  # Resize factor
INPUT_HEIGHT = ROI[3] // FACTOR
INPUT_WIDTH = ROI[2] // FACTOR
SPLIT_SEED = 42  # For train/val/test split
WEIGHTS_PTH = "cnn_model.pth"  # Path to the trained model

# Arrow keys
UP_KEY = 0 #82
DOWN_KEY = 1 #84
RIGHT_KEY = 3
LEFT_KEY = 2
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # S key
