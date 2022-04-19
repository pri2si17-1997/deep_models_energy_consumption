from model import *
from losses import *
import get_data
from data_generator import DataGenerator

import tensorflow as tf

_, _, X_test = get_data.get_data()

seg_model = get_model()

# compling model and callbacks functions
adam = tf.keras.optimizers.Adam(lr = 0.05, epsilon = 0.1)
seg_model.compile(optimizer = adam, 
                  loss = focal_tversky, 
                  metrics = [tversky, dice_coef]
                 )

# Change QAT weights here
seg_model.load_weights('../weights/ResUNet-segModel-weights.hdf5')

# Evaluaute model.
test_ids = list(X_test.image_path)
test_mask = list(X_test.mask_path)
test_data = DataGenerator(test_ids, test_mask)
_, tv, dice = seg_model.evaluate(test_data)

print("Segmentation tversky is {:.2f}%".format(tv*100))
print("Segmentation Dice is {:.2f}".format(dice))