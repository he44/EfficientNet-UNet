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


def efficientnet_unet(input_shape = (224,224,3), num_classes=1):
    encoder_model = get_efficientnet(name='B0', input_shape=input_shape)
    new_input = encoder_model.input
    # encoder output, we won't use the top_conv (which has 1280 filters)
    # let's just use 7a bn, which is 7 x 7 x 320
    encoder_output = encoder_model.get_layer(name='block7a_project_bn').output

    # filter number for the bottleneck
    fn_bottle_neck = encoder_output.shape[-1]
    bottleneck = CBAR_block(encoder_output, fn_bottle_neck)

    # Decoder block 1
    c1 = encoder_model.get_layer(name='block5c_drop').output
    fn_1 = c1.shape[-1]
    upsampling1 = tf.keras.layers.UpSampling2D()(bottleneck)
    concatenation1 = tf.keras.layers.concatenate(
            [upsampling1, c1], axis=3)
    decoder1 = CBAR_block(concatenation1, fn_1)

    # Decoder block 2
    c2 = encoder_model.get_layer(name='block3b_drop').output
    fn_2 = c2.shape[-1]
    upsampling2 = tf.keras.layers.UpSampling2D()(decoder1)
    concatenation2 = tf.keras.layers.concatenate(
            [upsampling2, c2], axis=3)
    decoder2 = CBAR_block(concatenation2, fn_2)

    # Decoder block 3
    c3 = encoder_model.get_layer(name='block2b_drop').output
    fn_3 = c3.shape[-1]
    upsampling3 = tf.keras.layers.UpSampling2D()(decoder2)
    concatenation3 = tf.keras.layers.concatenate(
            [upsampling3, c3], axis=3)
    decoder3 = CBAR_block(concatenation3, fn_3)

    # Decoder block 4
    # 1a does not have dropout
    c4 = encoder_model.get_layer(name='block1a_project_bn').output
    fn_4 = c4.shape[-1]
    upsampling4 = tf.keras.layers.UpSampling2D()(decoder3)
    concatenation4 = tf.keras.layers.concatenate(
            [upsampling4, c4], axis=3)
    decoder4 = CBAR_block(concatenation4, fn_4)

    # Decoder block 5
    # the only layer with original shape is input...
    fn_5 = fn_4 # let's resuse this filter number for now
    upsampling5 = tf.keras.layers.UpSampling2D()(decoder4)
    concatenation5 = tf.keras.layers.concatenate(
            [upsampling5, new_input], axis=3)
    decoder5 = CBAR_block(concatenation5, fn_5)

    # Now we can add in the output portion
    if num_classes == 1 or num_classes == 2:
        final_filter_num = 1
        final_activation = 'sigmoid'
    else:
        final_filter_num = num_classes
        final_activation = 'softmax'
    new_output = tf.keras.layers.Conv2D(filters=final_filter_num, kernel_size=1, activation=final_activation)(decoder5)

    print("output shape", new_output.shape)
    
    efficient_unet = tf.keras.Model(inputs=new_input, outputs=new_output)

    return efficient_unet

def main():
    model = efficientnet_unet(num_classes = 2)
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(), 
        optimizer = tf.keras.optimizers.Adam(1e-4)
    )
    model.summary()
    model.save('EU_Test.h5')

if __name__ == "__main__":
    main()
