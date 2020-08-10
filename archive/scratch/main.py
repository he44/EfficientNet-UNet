# Change the print info level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import efficientnet_builder
import numpy as np
import tensorflow as tf
print("Using TensorFlow v", tf.__version__)

images = np.zeros((1,224,224,3), dtype=np.float32)
images = tf.convert_to_tensor(images)
features, endpoints, model = efficientnet_builder.build_model_base(images, 'efficientnet-b0', training=True)


print(type(model))
model.summary()
model.save('test_b0.h5')