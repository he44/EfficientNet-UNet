import tensorflow as tf
print(tf.__version__)
import numpy as np
from keras_applications import efficientnet


def get_efficientnet(name='B0', input_shape=(None, None, 1)):
    return efficientnet.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=1,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils
    )


def efficientnet_unet(input_shape = (224,224,3)):
    encoder_model = get_efficientnet(name='B0', input_shape=input_shape)
    new_input = encoder_model.input
    new_output = encoder_model.output
    print(type(new_input), type(new_output))

    efficient_unet = tf.keras.Model(inputs=new_input, outputs=new_output)
    return efficient_unet

eunet = efficientnet_unet()
eunet.summary()


