import tensorflow as tf
print(tf.__version__)
import numpy as np
from keras_applications import efficientnet

# Convolution, Batch Normalization, Activation then Residual Connection
def CBAR_block(input, num_filters):
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    xd = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1)(input)
    x = tf.keras.layers.Add()([x, xd])

    return x


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
    encoder_output = encoder_model.output
    print(encoder_output.shape)

    bottleneck = CBAR_block(encoder_output, 16)

    print("5c shape", encoder_model.get_layer(name='block5c_drop').output.shape)

    upsampling1 = tf.keras.layers.UpSampling2D()(bottleneck)
    concatenation1 = tf.keras.layers.concatenate(
            [upsampling1, encoder_model.get_layer(name='block5c_drop').output], axis=3)


    new_output = concatenation1
    print(new_output.shape)
    
    efficient_unet = tf.keras.Model(inputs=new_input, outputs=new_output)

    return efficient_unet

def main():
    model = efficientnet_unet()
    model.summary()
    model.save('EU_Test.h5')

if __name__ == "__main__":
    main()
