import tensorflow as tf
print(tf.__version__)
import numpy as np
from keras_applications import efficientnet

model = efficientnet.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    classes=1,
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    utils=tf.keras.utils
)

model.summary()

model.save('EfficientNet-B0_keras.h5')

for layer in model.layers:
    print(type(layer))
