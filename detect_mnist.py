import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolo_core.models import Create_Yolo
from yolo_core.utils import detect_image
from configuration import *

while True:
    ID = random.randint(0, 200)
    label_txt = "./dataset/mnist/mnist_test.txt"
    image_info = open(label_txt).readlines()[ID].split()

    image_path = image_info[0]

    save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, DATA_TYPE)

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(save_directory) # use keras weights

    detect_image(yolo, image_path, "output_mnist.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
