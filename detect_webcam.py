import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from yolo_core.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from configuration import *

yolo = Load_Yolo_model()
detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))
