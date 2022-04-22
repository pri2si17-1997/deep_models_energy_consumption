from get_data import *
from metrics import MultipleClassAUROC
from model import *
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import tensorflow_model_optimization as tfmot

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope

class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []
    
    def get_activations_and_quantizers(self, layer):
        return []
    
    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
    num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

    def get_config(self):
        return {}
    
def apply_quantization_to_batch_normalization(layer):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        return quantize_annotate_layer(layer, DefaultBNQuantizeConfig())
    
    return layer

DATASET_PATH = os.path.abspath('/data/NIH_14')

data, all_labels = get_data(DATASET_PATH)
_, _, test_df = split_train_dev_test(data)
test_gen = get_data_generator(test_df, all_labels)

model = get_model(all_labels)

annotated_model = tf.keras.models.clone_model(
                    model,
                    clone_function=apply_quantization_to_batch_normalization,
)

with quantize_scope(
  {'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig}):
  # Use `quantize_apply` to actually make the model quantization aware.
  quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

initial_learning_rate=1e-3
optimizer = Adam(lr=initial_learning_rate)
quant_aware_model.compile(optimizer=optimizer, loss="binary_crossentropy")

quant_aware_model.load_weights('../weights/FP_32_QAT_weights.h5')

count = 0
while True:
    output = quant_aware_model.predict_generator(test_gen, steps=test_gen.n/test_gen.batch_size, verbose = True)
    print(f"Predicted Output : {output}")
    count += 1
    if count == 100000:
        break
