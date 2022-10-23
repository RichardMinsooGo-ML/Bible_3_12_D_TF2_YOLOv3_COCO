dataset_name = "mnist"
# dataset_name = "fashion_mnist"
# dataset_name = "voc"
# dataset_name = "coco"
# dataset_name = "OID_v6"

TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = True # "saved_model/yolov3_custom"
TRAIN_YOLO_TINY             = False

YOLO_CUSTOM_WEIGHTS         = True # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection    
# YOLO options
YOLO_TYPE                   = "yolov3" # yolov4 or yolov3
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"


YOLO_V3_WEIGHTS             = "./checkpoints/yolov3.weights"
YOLO_V3_TINY_WEIGHTS        = "./checkpoints/yolov3-tiny.weights"
YOLO_V4_WEIGHTS             = "./checkpoints/yolov4.weights"
YOLO_V4_TINY_WEIGHTS        = "./checkpoints/yolov4-tiny.weights"
# YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_COCO_CLASSES           = "./dataset/coco.names"
    
if dataset_name == "mnist":
    TRAIN_CLASSES               = "./dataset/mnist/mnist.names"
    TRAIN_ANNOT_PATH            = "./dataset/mnist/mnist_train.txt"
    TEST_ANNOT_PATH             = "./dataset/mnist/mnist_test.txt"
    if YOLO_TYPE == "yolov3":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v3_mnist"
    elif YOLO_TYPE == "yolov4":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v4_mnist"

elif dataset_name == "fashion_mnist":
    TRAIN_CLASSES               = "./dataset/fashion_mnist/mnist.names"
    TRAIN_ANNOT_PATH            = "./dataset/fashion_mnist/mnist_train.txt"
    TEST_ANNOT_PATH             = "./dataset/fashion_mnist/mnist_test.txt"
    if YOLO_TYPE == "yolov3":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v3_fashion_mnist"
    elif YOLO_TYPE == "yolov4":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v4_fashion_mnist"

elif dataset_name == "voc":
    TRAIN_CLASSES               = "./dataset/voc/voc2012.names"
    TRAIN_ANNOT_PATH            = "./dataset/voc/VOC2012_train.txt"
    TEST_ANNOT_PATH             = "./dataset/voc/VOC2012_val.txt"
    if YOLO_TYPE == "yolov3":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v3_voc"
    elif YOLO_TYPE == "yolov4":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v4_voc"
    
elif dataset_name == "coco":
    TRAIN_CLASSES               = "./dataset/coco/coco.names"
    TRAIN_ANNOT_PATH            = "./dataset/coco/COCO2017_train.txt"
    TEST_ANNOT_PATH             = "./dataset/coco/COCO2017_val.txt"
    if YOLO_TYPE == "yolov3":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v3_coco"
    elif YOLO_TYPE == "yolov4":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v4_coco"
        
elif dataset_name == "OID_v6":
    TRAIN_CLASSES               = "./dataset/OID_v6/OID_V6.names"
    TRAIN_ANNOT_PATH            = "./dataset/OID_v6/OID_V6_train.txt"
    TEST_ANNOT_PATH             = "./dataset/OID_v6/OID_V6_test.txt"
    if YOLO_TYPE == "yolov3":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v3_OID_v6"
    elif YOLO_TYPE == "yolov4":
        TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
        DATA_TYPE = "yolo_v4_OID_v6"

if TRAIN_YOLO_TINY:
    # TRAIN_CHECKPOINTS_FOLDER  += "_tiny"
    DATA_TYPE  += "_tiny"
    
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416

if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]

if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
# Train options
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOGDIR                = "log"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}"+"_"+dataset_name
TRAIN_LOAD_IMAGES_TO_RAM    = True # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 16
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 4

# TEST options
TEST_BATCH_SIZE             = 16
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

SIZE_TRAIN = 256*TRAIN_BATCH_SIZE
SIZE_TEST  = 64*TEST_BATCH_SIZE

#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]
