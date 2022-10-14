import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
# from yolo_core.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolo_core.utils import detect_video, Load_Yolo_model
from configuration import *

yolo = Load_Yolo_model()

# video_path   = "./IMAGES/street.mp4"
# detect_video(yolo, video_path, "./pred_IMAGES/Pred_street.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))

# video_path   = "./IMAGES/Highway.mp4"
# detect_video(yolo, video_path, "./pred_IMAGES/Pred_Highway.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))

video_path   = "./IMAGES/street.mp4"
detect_video(yolo, video_path, "./pred_IMAGES/pred_street.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))

