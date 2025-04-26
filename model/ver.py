from yolo_loss import IoU, cell_loss
import tensorflow as tf


# def debug_one():
#     S   = 7
#     idx = 24                       # middle cell
#     # construct a perfect box centred at cell (3,3)
#     row, col = idx // S, idx % S
#     cx = (col + 0.5) / S
#     cy = (row + 0.5) / S
#     w  = h = 1.0 / S               # one-cell-wide square

#     # build ground-truth vector  (tx,ty,√w,√h, one-hot[person])
#     tx = 0.5;  ty = 0.5
#     truth = tf.concat([
#         tf.constant([tx, ty, tf.sqrt(w), tf.sqrt(h)]),
#         tf.one_hot(14, 20, dtype=tf.float32)      # “person”
#     ], axis=0)

#     # build pred head 1 identical to ground truth (after activations)
#     pred_head = tf.concat([
#         tf.constant([0.5, 0.5, tf.sqrt(w), tf.sqrt(h), 1.0]),   # p1
#         tf.zeros(5),                                            # p2 dummy
#         tf.zeros(20)                                            # class logits
#     ], axis=0)

#     loss = _cell_loss(pred_head, truth, idx, S)
#     print("coord part should be 0, total loss =", loss.numpy())

# debug_one()
eps = 1e-6

import tensorflow as tf
from yolo_loss import cell_loss

def safe_sigmoid_inv(y, eps=1e-6):
    y = tf.clip_by_value(y, eps, 1 - eps)
    return tf.math.log(y / (1 - y))

def test_cell_loss_fixed():
    S = 7
    idx = 24
    row = idx // S
    col = idx % S

    # Desired center and size in absolute coords
    cx = (col + 0.5) / S
    cy = (row + 0.5) / S
    w = h = 1.0 / S

    # Inverse activations to get raw input values that sigmoid/square will match
    x_raw = safe_sigmoid_inv(0.5)  # gives offset 0.5 in cell
    y_raw = safe_sigmoid_inv(0.5)
    w_raw = tf.sqrt(tf.constant(w, dtype=tf.float32))  # predicts sqrt(w)
    h_raw = tf.sqrt(tf.constant(h, dtype=tf.float32))
    c_raw = safe_sigmoid_inv(1.0)  # perfect confidence

    # Prediction vector (30 dim)
    pred = tf.concat([
        tf.stack([x_raw, y_raw, w_raw, h_raw, c_raw]),       # head 1
        tf.zeros(5),                                         # head 2
        tf.zeros(20)                                         # logits
    ], axis=0)

    # Ground-truth box in [xmin, ymin, xmax, ymax, class_id]
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2
    gt = tf.constant([xmin, ymin, xmax, ymax, 0.0], dtype=tf.float32)

    # Evaluate the loss
    loss_val = cell_loss(pred, gt, idx=idx, S=S)
    print(f"Perfect match → cell_loss = {loss_val.numpy():.10f}")

test_cell_loss_fixed()


# Sanity 1 – IoU should be 1 for identical boxes
b = tf.constant([.5, .5, .2, .3])             # cx,cy,w,h
g = tf.constant([.4, .35, .6, .65, 0])        # xmin,ymin,xmax,ymax
assert IoU(b, g).numpy() > 0.99

# Sanity 2 – cell_loss should be small for perfect prediction
dummy_pred = tf.concat([b, [1.0], b, [0.]], axis=0)  # two heads
dummy_gt   = g                                       # same box
print(cell_loss(dummy_pred, dummy_gt, idx=24, S=7).numpy())


# import tensorflow as tf
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# # VOC class names in index order
# VOC_CLASSES = [
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow',
#     'diningtable', 'dog', 'horse', 'motorbike', 'person',
#     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]

# def visualize_sample(model=None, data_dir='~/tensorflow_datasets', target_size=448, grid_size=7):
#     """
#     Fetch one sample from the VOC2007 training set, preprocess it,
#     display the image with ground-truth bounding boxes (green) and labels,
#     and if a model is provided, overlay predicted boxes and classes (red).
#     """
#     # Load and preprocess one sample
#     ds = (
#         tfds.load('voc/2007', split='train', data_dir=data_dir)
#         .map(preprocess, tf.data.AUTOTUNE)
#     )
#     image, y_true = next(iter(ds.take(1)))
    
#     fig, ax = plt.subplots(figsize=(6,6))
#     ax.imshow(image.numpy())
#     ax.axis('off')

#     # Draw ground-truth boxes
#     for i in range(grid_size):
#         for j in range(grid_size):
#             cell = y_true[i, j]
#             if tf.reduce_any(cell[:4] > 0):
#                 xmin, ymin, xmax, ymax = cell[:4]
#                 cls_id = tf.argmax(cell[4:]).numpy()
#                 # to pixel coordinates
#                 x1, y1 = xmin * target_size, ymin * target_size
#                 w, h = (xmax - xmin) * target_size, (ymax - ymin) * target_size
#                 rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', facecolor='none')
#                 ax.add_patch(rect)
#                 ax.text(x1, y1 - 2, VOC_CLASSES[cls_id], color='green', fontsize=8, backgroundcolor='white')
    
#     # If a model is provided, draw predictions
#     if model is not None:
#         pred = model(tf.expand_dims(image, 0))[0]  # (7,7,30)
#         flat = tf.reshape(pred, [-1, 30]).numpy()
#         for idx, out in enumerate(flat):
#             # choose best of the 2 boxes
#             if out[4] >= out[9]:
#                 x_raw, y_raw, w_raw, h_raw, conf = out[:5]
#             else:
#                 x_raw, y_raw, w_raw, h_raw, conf = out[5:10]
#             if conf < 0.2:
#                 continue
#             row = idx // grid_size
#             col = idx % grid_size
#             # convert to normalized absolute
#             cx = (col + x_raw) / grid_size
#             cy = (row + y_raw) / grid_size
#             w_abs = w_raw
#             h_abs = h_raw
#             xmin = cx - w_abs/2
#             ymin = cy - h_abs/2
#             xmax = cx + w_abs/2
#             ymax = cy + h_abs/2
#             # pixel coords
#             x1, y1 = xmin * target_size, ymin * target_size
#             w_pix, h_pix = (xmax - xmin) * target_size, (ymax - ymin) * target_size
#             cls_id = np.argmax(out[10:])
#             rect = Rectangle((x1, y1), w_pix, h_pix, linewidth=1, edgecolor='red', facecolor='none')
#             ax.add_patch(rect)
#             ax.text(x1, y1 - 2, VOC_CLASSES[cls_id], color='red', fontsize=8, backgroundcolor='white')
    
#     plt.show()

# # Example usage:
# # visualize_sample(model=yolov1)
