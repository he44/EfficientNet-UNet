# U-Net with EfficientNet as Encoder

## Reference

### Original Paper:

- [U-Net](https://arxiv.org/abs/1505.04597)

- [Efficient Net](https://arxiv.org/abs/1905.11946)

### Existing implementation

- [EfficientUnet](https://github.com/zhoudaxia233/EfficientUnet)

- [keras-EfficientNet-Unet](https://github.com/madara-tribe/keras-EfficientNet-Unet)

- [EfficientUnet-PyTorch](https://github.com/zhoudaxia233/EfficientUnet-PyTorch)

- [Efficient Net from the original author](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Environment

- CUDA 11.0

- Anaconda: Python 3.7

- Tensorflow 2.1.0

## EfficientNet B0

### Architecture

   ![B0](B0.png)
   
## Files and directory

### scratch

Trying to build the pre-trained network following [the repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) linked in the paper. 

## Notes:

- somehow Tensorflow 2.2 won't work with the given code. There's an error message saying "tensor.name is meaningless in eager execution". 

   ```Python
   features, endpoints = efficientnet_builder.build_model_base(images, model_name = 'efficientnet-b0', training=False)
   ```

   Quickly checked [stackoverlfow](https://stackoverflow.com/questions/52340101/tensor-name-is-meaningless-in-eager-execution), didn't find a very executable solution except for using TF 1.x. For the sake of time, I chagned to TF 1.15.

   @TODO: figuring out why 2.0 won't work

- The images should have 4 dimensions in the code above.

   @TODO: why does 1 channel tensor work too? I thought it's expecting RGB?

- 