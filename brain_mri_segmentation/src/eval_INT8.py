import os
from model import *
from losses import *
import get_data
from data_generator import DataGenerator

import tensorflow as tf
import tensorflow.keras.backend as K

def eval_tflite(interpreter, test_data, n):
    mean_dice = 0
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (16, 256, 256, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (16, 256, 256, 1))
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    print(f"Input Details : {input_details}")
    print(f"Output Details : {output_details}")
    interpreter.allocate_tensors()
    for x, y in test_data:
        x_img = x.astype(input_details["dtype"])
        print(x_img.dtype)
        interpreter.set_tensor(input_details['index'], x_img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        dice_score = K.eval(dice_coef(y.astype('float32'), output_data.astype('float32')))
        mean_dice += dice_score
    
    return mean_dice / n
    

def eval_int8(test_data, n):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_Brain_MRI_Segmentation.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH)
    return eval_tflite(interpreter, test_data, n)
    

def eval_int8_edge_tpu(test_data, n):
    INT8_MODEL_PATH = os.path.abspath('../weights/QAT_INT8_Brain_MRI_Segmentation_edgetpu.tflite')
    interpreter = tf.lite.Interpreter(model_path = INT8_MODEL_PATH, experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')])
    return eval_tflite(interpreter, test_data, n)


if __name__ == "__main__":
    _, _, X_test = get_data.get_data()

    test_ids = list(X_test.image_path)
    test_mask = list(X_test.mask_path)
    test_data = DataGenerator(test_ids, test_mask)

    int8_dice_score = eval_int8(test_data, len(test_data))
    print(f"Dice Score for INT8 Model : {int8_dice_score}")

    try:
        int8_dice_score_edge = eval_int8_edge_tpu(test_data, len(test_data))
        print(f"Dice Score for INT8 Model on edge TPU: {int8_dice_score_edge}")
    except Exception as ex:
        print(f"Not supported edge TPU found.")
