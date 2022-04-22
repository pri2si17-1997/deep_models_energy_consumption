from matplotlib.pyplot import get
from get_data import *
from utils import *
import numpy as np

try:
    import tensorflow as tf
except Exception:
    import tflite_runtime.interpreter as tflite
    
import os
import sys

def get_every_n(a, n=2):
    for i in range(a.shape[0] // n):
        yield a[n*i:n*(i+1)]

def eval_tflite(interpreter, imgs_test):
    mean_dice = 0
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (32, 80, 112, 1))
    interpreter.resize_tensor_input(output_details[0]['index'], (32, 80, 112, 1))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input Details : {input_details}")
    print(f"Output Details : {output_details}")
    interpreter.allocate_tensors()

    for data in get_every_n(imgs_test, 32):
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Output Data : {output_data}")
    

def eval_int8(imgs_test):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_Nerve_Segmentation.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH)
    return eval_tflite(interpreter, imgs_test)
    

def eval_int8_edge_tpu(imgs_test):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_Nerve_Segmentation_edgetpu.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH, experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')])
    return eval_tflite(interpreter, imgs_test)

if __name__ == "__main__":
    print(f"PID : {os.getpid()}")
    IMG_ROWS, IMG_COLS = 80, 112
    imgs_test = load_test_data()
    imgs_test = preprocess(imgs_test, IMG_ROWS, IMG_COLS)
    imgs_test = imgs_test.astype('float32')
    
    MEAN = 98.06 #Calculated from train data.
    STD = 51.57 #Calculated from train data.

    imgs_test -= MEAN
    imgs_test /= STD

    if int(sys.argv[1]) == 0:
        print(f"Processing INT8 Model..")
        count = 0
        while True:
            eval_int8(imgs_test=imgs_test)
            count += 1
            if count == 100000:
                break
                
    elif int(sys.argv[1]) == 1:
        print(f"Processing Edge TPU Compiled INT8 Model..")
        count = 0
        while True:
            try:
                eval_int8_edge_tpu(imgs_test=imgs_test)
                count += 1
                if count == 100000:
                    break
            except Exception:
                print(f"Edge TPU not found.")
