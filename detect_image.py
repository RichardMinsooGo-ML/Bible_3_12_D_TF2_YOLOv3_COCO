import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
# from yolo_core.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolo_core.utils import detect_image, Load_Yolo_model
from configuration import *

yolo = Load_Yolo_model()

image_path   = "./IMAGES/kite.jpg"
detect_image(yolo, image_path, "./pred_IMAGES/Pred_kite.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))

image_path   = "./IMAGES/dog.jpg"
detect_image(yolo, image_path, "./pred_IMAGES/Pred_dog.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))

image_path   = "./IMAGES/city.jpg"
detect_image(yolo, image_path, "./pred_IMAGES/Pred_city.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))

image_path   = "./IMAGES/street.jpg"
detect_image(yolo, image_path, "./pred_IMAGES/Pred_street.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))

