from tensorflow.keras.layers import Input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

def get_model(all_labels):
    input_shape=(224, 224, 3)
    img_input = Input(shape=input_shape)

    base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, 
                         pooling="avg", weights='imagenet')
    x = base_model.output
    predictions = Dense(len(all_labels), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)

    return model