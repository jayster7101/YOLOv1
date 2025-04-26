## Load Dataset ##
import os

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

# Currently broken, must resize the bbox as they are being converted to pixel size, but then need to scale bbox accordingly to maintain relative scaling 
def resize_and_rescale(example):
    image = example['image']
    bboxes = example['objects']['bbox']  # [ymin, xmin, ymax, xmax], normalized
    labels = example['objects']['label']  # class indices

    # print("Image", image)
    # print("bboxes", bboxes)
    # print("labels", labels)

    # Since Normalized, no need to scale since they will scale proportionally lresru 
    # shape = tf.shape(image)
    # x,y,channels = tf.unstack(shape) 
    # print("Input to resize and rescale value", example)
    # print("og dims", x,y,channels)
    # x_scale = TARGET_SIZE[0]/x
    # y_scale = TARGET_SIZE[1]/y
    ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis = -1)
    # ymin = y_scale * ymin
    # xmin = x_scale * xmin
    # ymax = y_scale * ymax
    # xmax = x_scale * xmax
    bboxes = tf.stack((xmin, ymin, xmax, ymax), axis=-1) # in correct order

    # Resize image to target size
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0, 1]

    class_ids = tf.cast(tf.expand_dims(labels, axis=1), tf.float32)  # shape (N, 1)

    # Combine into [xmin, ymin, xmax, ymax, class_id]
    final_targets = tf.concat([bboxes, class_ids], axis=1)  # shape (N, 5)

    return image, final_targets

def prep_for_inference(image_file_path):
    if not os.path.isfile(image_file_path):
        raise FileNotFoundError(f"Image not found: {image_file_path}")
      
    # Load + decode
    
    raw_img = tf.io.read_file(image_file_path)
    decoded_img = tf.image.decode_jpeg(raw_img, channels=3)
    print("Original decoded shape:", decoded_img.shape)

    # Normalize to [0,1] and resize
    prepped_img = tf.image.convert_image_dtype(decoded_img, tf.float32)  # [0,1]
    img = tf.image.resize(prepped_img, TARGET_SIZE)

    return img

    


def get_Data():
    builder = tfds.builder(
        'voc/2007',
        data_dir='/cs/student/jaydenjardine/tensorflow_datasets'
    )

    # Only run if dataset hasn't been prepared
    if not builder.info.splits:
        builder.download_and_prepare(
            download_config=tfds.download.DownloadConfig(
                manual_dir='/cs/student/jaydenjardine/tensorflow_datasets/downloads/manual'
            )
        )

    return builder.as_dataset()


# def voc_ds_gather():
#   ds_list = tfds.list_builders()
#   data_set_name = 'voc/2007'
#   if data_set_name not in ds_list and data_set_name  != 'voc/2007' :
#     print(ds_list)
#     print("You may have meant one of the following:")
#     print(get_close_matches(data_set_name,ds_list))
#     raise ValueError(f"Data set {data_set_name} not apart of Tensorflows Dataset library.")

#   ds, info = tfds.load(data_set_name, split='train', with_info=True)
#   assert isinstance(ds, tf.data.Dataset)
#   print(ds,info)
#   return ds, info




# # Get just one example
# for example in ds.take(1):
#     print(example)

#     for content in example:
#        print("Content", content)
# # Get just one example
# for example in dataset.take(1):
#     print("modifed")
#     print(example)
