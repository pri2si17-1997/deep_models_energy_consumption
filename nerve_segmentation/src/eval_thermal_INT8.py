from get_data import *
from utils import *
import numpy as np

try:
    import tensorflow as tf
except Exception:
    import tflite_runtime.interpreter as tflite

def eval_tflite(interpreter, imgs_test):
    mean_dice = 0
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (5508, 80, 112, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (5508, 80, 112, 1))
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    print(f"Input Details : {input_details}")
    print(f"Output Details : {output_details}")
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], imgs_test)
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])
    

def eval_int8(imgs_test):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_Nerve_Segmentation.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH)
    return eval_tflite(interpreter, imgs_test)
    

def eval_int8_edge_tpu(imgs_test):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_Nerve_Segmentation_edgetpu.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH, experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')])
    return eval_tflite(interpreter, imgs_test)

if __name__ == "__main__":
    IMG_ROWS, IMG_COLS = 80, 112
    imgs_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_test = imgs_test.astype('float32')
    MEAN = 98.06 #Calculated from train data.
    STD = 51.57 #Calculated from train data.

    imgs_test -= MEAN
    imgs_test /= STD

    count = 0
    while True:
        eval_int8(imgs_test=imgs_test)
        count += 1
        if count == 100000:
            break
