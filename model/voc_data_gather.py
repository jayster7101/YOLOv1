## Load Dataset ##

'''
This file gathers the data
  - images
  - labels
This file also scales images to the expected model input size, in this model thats 448
  - must also scale the labels to this size
  - we then proceed forward with all data in its normalized format ie ~0-1

  **!TO DO!**
    - 1 get data in correct format
      - scale image size to 448x448 (parameterized)
      - scale labels accordingly weight by relative scaling factor
      - normalize ~[0,1]

    - 2, in training, ensure that we can do batching, there is a command that allows us to do that

    - 3, if training takes some time, the loss function has a clear path for pure parallelized implementation,
      - swap out python logic for tf logic, calling of loss via the tf reduce sum, but would have to figure out how to know indices or cell mapping 

  

'''



import tensorflow as tf
import tensorflow_datasets as tfds
from difflib import get_close_matches

TARGET_SIZE = (448, 448)

def resize_and_rescale(example):
    image = example['image']
    bboxes = example['objects']['bbox']  # [ymin, xmin, ymax, xmax], normalized
    labels = example['objects']['label']  # class indices

    # Resize image to target size
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0, 1]

    # Convert normalized bboxes to pixel values in [xmin, ymin, xmax, ymax] format
    ymin = bboxes[:, 0] * TARGET_SIZE[0]
    xmin = bboxes[:, 1] * TARGET_SIZE[1]
    ymax = bboxes[:, 2] * TARGET_SIZE[0]
    xmax = bboxes[:, 3] * TARGET_SIZE[1]

    pixel_bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)
    class_ids = tf.cast(tf.expand_dims(labels, axis=1), tf.float32)  # shape (N, 1)

    # Combine into [xmin, ymin, xmax, ymax, class_id]
    final_targets = tf.concat([pixel_bboxes, class_ids], axis=1)  # shape (N, 5)
    # # Combine with class labels if needed later
    # example['image'] = image
    # example['objects']['bbox'] = pixel_bboxes  # now in pixel format [xmin, ymin, xmax, ymax]

    return image, final_targets


def voc_ds_gather():
  ds_list = tfds.list_builders()
  data_set_name = 'voc'
  if data_set_name not in ds_list and data_set_name  != 'voc2007' :
    print(ds_list)
    print("You may have meant one of the following:")
    print(get_close_matches(data_set_name,ds_list))
    raise ValueError(f"Data set {data_set_name} not apart of Tensorflows Dataset library.")

  ds, info = tfds.load(data_set_name, split='train', with_info=True)
  assert isinstance(ds, tf.data.Dataset)
  print(ds,info)
  return ds, info




# # Get just one example
# for example in ds.take(1):
#     print(example)

#     for content in example:
#        print("Content", content)
# # Get just one example
# for example in dataset.take(1):
#     print("modifed")
#     print(example)
