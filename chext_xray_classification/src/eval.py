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
pred_y = model.predict_generator(test_gen, steps=test_gen.n/test_gen.batch_size, verbose = True)


test_gen.reset()
test_x, test_y = next(test_gen)

# Space
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    #Points to graph
    fpr, tpr, thresholds = roc_curve(test_gen.labels[:,idx].astype(int), pred_y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
#convention
c_ax.legend()

#Labels
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')

# Save as a png
fig.savefig('QAT_FP32.png')