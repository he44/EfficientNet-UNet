import tensorflow_datasets as tfds

train_coco, test_coco = tfds.load(
    'coco', split=['train', 'test'], 
    shuffle_files=True,
    as_supervised=True
)

print(type(train_coco, test_coco))
