import efficientnet_builder
import numpy as np
import tensorflow as tf
images = np.ones((1,224,224,3))
images = tf.constant(images)
features, endpoints = efficientnet_builder.build_model_base(images, 'efficientnet-b0', training=False)