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
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils
    )


def efficientnet_unet(input_shape = (224,224,3)):
    encoder_model = get_efficientnet(name='B0', input_shape=input_shape)
    new_input = encoder_model.input
    new_output = encoder_model.output
    efficient_unet = tf.keras.Model(inputs=new_input, outputs=new_output)
    return efficient_unet

#eunet.summary()


#input_name = 'panda'
input_name = 'labrador'
import imageio
images = imageio.imread("%s.jpg"%input_name)
images = np.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))
images = images.astype(np.float32)
print(images.shape, images.dtype)


eunet = efficientnet_unet(input_shape=images.shape[1:])

layer_3b = eunet.get_layer(name='block3b_project_bn')
inter_model = tf.keras.Model(inputs = eunet.input, outputs = layer_3b.output)
activation_maps = inter_model.predict(images)


import matplotlib.pyplot as plt
import os
import random

dev_tmp = r'./../tmp'
ks = random.sample(range(activation_maps.shape[-1]), 10)
for k in ks:
    plt.imshow(activation_maps[0, :, :, k])
    plt.savefig(os.path.join(dev_tmp, 'keras_3b_channel_%d__%s.png'%(k, input_name)))
    plt.clf()

