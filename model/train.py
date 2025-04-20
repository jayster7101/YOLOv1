import tensorflow as tf
import tensorflow_datasets as tfds

from voc_data_gather import voc_ds_gather, resize_and_rescale

from yolov1 import YoloV1_

from yolo_loss import loss_fn

# Collect dataset and scale accordingly 
ds, info = voc_ds_gather()

dataset = ds.map(resize_and_rescale, num_parallel_calls=tf.data.AUTOTUNE)

# for example in dataset.take(1):
#     print("modifed")
#     print(example)


# dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

yolov1 = YoloV1_(input_size=448)


yolov1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn
)

for image, labels in dataset.take(1):
    y_pred = yolov1(image)
    loss_val = loss_fn(y_pred, labels)
    print("Loss:", loss_val.numpy())
