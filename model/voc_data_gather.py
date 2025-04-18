## Load Dataset ##
import tensorflow as tf
import tensorflow_datasets as tfds
from difflib import get_close_matches


ds_list = tfds.list_builders()
data_set_name = 'voc/2012'
if data_set_name not in ds_list and data_set_name  != 'voc/2012' :
  print(ds_list)
  print("You may have meant one of the following:")
  print(get_close_matches(data_set_name,ds_list))
  raise ValueError(f"Data set {data_set_name} not apart of Tensorflows Dataset library.")


builder = tfds.builder('voc/2012')
builder.download_and_prepare(download_dir='~/tensorflow_datasets/downloads/')
filename = data_set_name + '.tfrecord'
ds, info = tfds.load('voc/2012', split='train', with_info=True, download=False)

assert isinstance(ds, tf.data.Dataset)
print(ds)

