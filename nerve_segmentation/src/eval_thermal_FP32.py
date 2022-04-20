import tensorflow as tf
from tensorflow.keras import backend as K

from get_data import *
from losses import *
from model import *
from utils import *
import numpy as np

IMG_ROWS, IMG_COLS = 80, 112
K.set_image_data_format('channels_last')

unet_model = get_unet_customised(Adam(lr = 1e-5), pars=PARS, allowed_pars=ALLOWED_PARS, IMG_ROWS=IMG_ROWS, IMG_COLS=IMG_COLS)
q_aware_unet_model = get_QAT_model(unet_model)

mean = 98.06 #Calculated from train data.
std = 51.57 #Calculated from train data.

print('-' * 30)
print('Loading and preprocessing test data...')
print('-' * 30)
imgs_test = load_test_data()
imgs_test = preprocess(imgs_test)

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-' * 30)
print('Loading saved weights...')
print('-' * 30)
q_aware_unet_model.load_weights('../weights/QAT_INT8_Nerve_Segmentation.tflite')

print('-' * 30)
print('Predicting masks on test data...')
print('-' * 30)

count = 0
while True:
    q_aware_unet_model.predict(imgs_test, verbose=1)
    count += 1

    if count == 100000:
        break
