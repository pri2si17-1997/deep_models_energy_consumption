from get_data import *
from metrics import MultipleClassAUROC
from model import *
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

DATASET_PATH = os.path.abspath('../dataset/')

data, all_labels = get_data(DATASET_PATH)
_, _, test_df = split_train_dev_test(data)
test_gen = get_data_generator(test_df, all_labels)

model = get_model()

initial_learning_rate=1e-3
optimizer = Adam(lr=initial_learning_rate)
model.compile(optimizer=optimizer, loss="binary_crossentropy")

model.load_weights('../weights/FP_32_QAT_weights.h5')

count = 0
while True:
    model.predict_generator(test_gen, steps=test_gen.n/test_gen.batch_size, verbose = True)
    count += 1
    if count == 100000:
        break
