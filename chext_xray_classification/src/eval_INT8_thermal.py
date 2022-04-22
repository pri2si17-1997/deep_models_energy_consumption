import os

from get_data import *
import tensorflow as tf
import tensorflow.keras.backend as K
import sys


def eval_tflite(interpreter, test_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (32, 224, 224, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (32, 14))
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    print(f"Input Details : {input_details}")
    print(f"Output Details : {output_details}")
    interpreter.allocate_tensors()
    test_data.reset()
    for x, y in test_data:
        x_img = x.astype(input_details["dtype"])
        interpreter.set_tensor(input_details['index'], x_img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Output Data : {output_data}")

def eval_int8(test_data):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH)
    return eval_tflite(interpreter, test_data)
    

def eval_int8_edge_tpu(test_data, n):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_edgetpu.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH, experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')])
    return eval_tflite(interpreter, test_data, n)


if __name__ == "__main__":
    print(f"PID : {os.getpid()}")
    DATASET_PATH = os.path.abspath('/data/NIH_14')

    data, all_labels = get_data(DATASET_PATH)
    _, _, test_df = split_train_dev_test(data)
    test_gen = get_data_generator(test_df, all_labels)
    
    if int(sys.argv[1]) == 0:
        print(f"Evaluating INT 8 Model : ")
        count = 0
        while True:
            eval_int8(test_gen)
            count += 1
            if count == 100000:
                break
            
    elif int(sys.argv[1]) == 1:
        print(f"Evaluating Edge TPU Compiled INT8 Model : ")
        count = 0
        while True:
            eval_int8_edge_tpu(test_gen)
            count += 1
            if count == 100000:
                break
        