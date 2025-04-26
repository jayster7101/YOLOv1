# yolov1_fast.py  – accelerated YOLOv1 training loop
# Author: Jayden  (ChatGPT refactor help)
# ---------------------------------------------------------------------------
#  ✱ Vectorised loss   ✱ tf.data cache→shuffle→batch→prefetch
#  ✱ XLA ready         ✱ Mixed precision switch (commented)
# ---------------------------------------------------------------------------

import tensorflow as tf 
import tensorflow_datasets as tfds


# ───────────────────────────────────────────────────────── Environment ───────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # tf.config.optimizer.set_jit(True)               # XLA
# from tensorflow.keras import mixed_precision as mp
# mp.set_global_policy('mixed_float16')         # ← optional

AUTOTUNE    = tf.data.AUTOTUNE
TARGET_SIZE = (448, 448)
S, B, C     = 7, 2, 20                          # grid, boxes / cell, classes
L_COORD     = 5.0
L_NOOBJ     = 0.5
BATCH_SIZE  = 2                               # tune for your VRAM

EPS = 1e-7

# ───────────────────────────────────────────────────────── Layers / Model ────
class Conv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, strides=1, padding='same',
                 activation='lrelu', alpha=0.1):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel, strides=strides,
                                           padding=padding, use_bias=False,
                                           kernel_initializer='he_normal')
        if activation == 'lrelu':
            self.act = tf.keras.layers.LeakyReLU(alpha)
        elif activation == 'relu':
            self.act = tf.keras.layers.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def call(self, x):
        return self.act(self.conv(x))


def YoloV1_(input_size=448):
    inputs = tf.keras.Input([input_size, input_size, 3])
    x = inputs

    x = Conv2d(64, 7, 2)(x)
    x = Conv2d(64, 7)(x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)

    x = Conv2d(192, 3)(x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)

    x = Conv2d(128, 1)(x)
    x = Conv2d(256, 3)(x)
    x = Conv2d(256, 1)(x)
    x = Conv2d(512, 3)(x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)

    for _ in range(4):
        x = Conv2d(256, 1)(x)
        x = Conv2d(512, 3)(x)

    x = Conv2d(512, 1)(x)
    x = Conv2d(1024, 3)(x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)

    x = Conv2d(512, 1)(x)
    x = Conv2d(1024, 3)(x)
    x = Conv2d(512, 1)(x)
    x = Conv2d(1024, 3)(x)
    x = Conv2d(1024, 3)(x)
    x = Conv2d(1024, 3, 2)(x)
    x = Conv2d(1024, 3)(x)
    x = Conv2d(1024, 3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(S * S * (B * 5 + C))(x)
    outputs = tf.keras.layers.Reshape((S, S, B * 5 + C))(x)
    return tf.keras.Model(inputs, outputs)

# ───────────────────────────────────────────────────────── Loss helpers ──────
def corners_to_center(xmin, ymin, xmax, ymax):
    w = xmax - xmin
    h = ymax - ymin
    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h
    return xc, yc, w, h


def center_to_corners(xc, yc, w, h):
    xmin = xc - 0.5 * w
    ymin = yc - 0.5 * h
    xmax = xc + 0.5 * w
    ymax = yc + 0.5 * h
    return xmin, ymin, xmax, ymax


def bbox_iou(pred_xywh, gt_xyxy):
    """pred_xywh: (...,4) center‑form,  gt_xyxy: (...,4) corner‑form"""
    xmin1, ymin1, xmax1, ymax1 = center_to_corners(*tf.unstack(pred_xywh, axis=-1))
    xmin2, ymin2, xmax2, ymax2 = tf.unstack(gt_xyxy, axis=-1)

    inter_xmin = tf.maximum(xmin1, xmin2)
    inter_ymin = tf.maximum(ymin1, ymin2)
    inter_xmax = tf.minimum(xmax1, xmax2)
    inter_ymax = tf.minimum(ymax1, ymax2)

    inter_w = tf.maximum(inter_xmax - inter_xmin, 0.0)
    inter_h = tf.maximum(inter_ymax - inter_ymin, 0.0)
    inter_area = inter_w * inter_h

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = area1 + area2 - inter_area + EPS
    return inter_area / union


# ───────────────────────────────────────────────────────── Loss (vectorised) ─
# @tf.function(jit_compile=True)
def yolo_v1_loss(y_true, y_pred):  # shapes: (B,S,S,24)  (B,S,S,30)
    # Split prediction
    pred_boxes = tf.reshape(y_pred[..., :B * 5], (-1, S, S, B, 5))  # (...,B,5)
    pred_xy    = pred_boxes[..., 0:2]
    pred_wh    = pred_boxes[..., 2:4]
    pred_obj   = pred_boxes[..., 4]                                 # (...,B)
    pred_cls   = y_pred[..., B * 5:]                                # (B,S,S,C)

    # Ground truth
    gt_xyxy = y_true[..., :4]                                       # corners
    gt_cls  = y_true[..., 4:]
    obj_mask = tf.cast(tf.reduce_sum(gt_xyxy, axis=-1) > 0, tf.float32)  # (B,S,S)

    # IoU each predicted box vs gt in same cell
    gt_broadcast = tf.expand_dims(gt_xyxy, axis=3)                  # (B,S,S,1,4)
    pred_xywh    = tf.concat([pred_xy, pred_wh], axis=-1)           # (B,S,S,B,4)

    ious = bbox_iou(pred_xywh, gt_broadcast)                        # (B,S,S,B)

    # Responsible box mask (broadcast‑safe)
    best_idx  = tf.argmax(ious, axis=-1)                            # (B,S,S)
    best_box  = tf.one_hot(best_idx, depth=B, dtype=tf.float32)     # (B,S,S,B)
    obj_mask4 = tf.expand_dims(obj_mask, axis=-1)                   # (B,S,S,1)
    best_box  = best_box * obj_mask4                                # mask cells w/out obj

    # ── 1. localisation ──────────────────────────────────────────────
    gt_xywh = tf.stack(corners_to_center(*tf.unstack(gt_xyxy, axis=-1)),
                       axis=-1)                                     # (B,S,S,4)
    gt_xywh = tf.expand_dims(gt_xywh, axis=3)                       # (B,S,S,1,4)

    xy_loss = tf.square(pred_xy - gt_xywh[..., 0:2])
    wh_loss = tf.square(tf.sqrt(tf.maximum(pred_wh, EPS)) -
                        tf.sqrt(gt_xywh[..., 2:4]))

    coord_loss = tf.reduce_sum(best_box[..., tf.newaxis] *
                               (xy_loss + wh_loss), axis=[1, 2, 3, 4])

    # ── 2. object confidence ─────────────────────────────────────────
    obj_loss = tf.reduce_sum(best_box *
                             tf.square(pred_obj - ious), axis=[1, 2, 3])

    # ── 3. no‑object confidence ──────────────────────────────────────
    noobj_loss = tf.reduce_sum((1.0 - best_box) *
                               tf.square(pred_obj), axis=[1, 2, 3])

    # ── 4. class probability ─────────────────────────────────────────
    class_loss = tf.reduce_sum(obj_mask4 * tf.square(pred_cls - gt_cls), axis=[1, 2, 3])

    total = (L_COORD * coord_loss +
             obj_loss +
             L_NOOBJ * noobj_loss +
             class_loss)

    return tf.reduce_mean(total)    # scalar

# ───────────────────────────────────────────────────────── Data pipeline ─────
def make_truth_tensor(truth_boxes):
    """
    truth_boxes: (N,5)  → each row [xmin, ymin, xmax, ymax, class_id]
    returns      : (S,S, 4+C) with one object max per cell (remaining zeros)
    """
    if tf.shape(truth_boxes)[0] == 0:
        return tf.zeros((S, S, 4 + C), tf.float32)

    xmin, ymin, xmax, ymax, cls = tf.split(truth_boxes, [1, 1, 1, 1, 1], axis=1)
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    cell_x = tf.cast(x_center * S, tf.int32)
    cell_y = tf.cast(y_center * S, tf.int32)
    idx    = cell_y * S + cell_x                                  # (N,1)

    class_onehot = tf.one_hot(tf.cast(cls[:, 0], tf.int32), C)
    full_vec     = tf.concat([xmin, ymin, xmax, ymax, class_onehot], 1)

    flat = tf.tensor_scatter_nd_update(
        tf.zeros((S * S, 4 + C), tf.float32),
        idx,
        full_vec
    )
    return tf.reshape(flat, (S, S, 4 + C))


def preprocess(example):
    img   = tf.image.resize(example['image'], TARGET_SIZE)
    img   = tf.cast(img, tf.float32) / 255.0

    # VOC bbox order: [ymin,xmin,ymax,xmax] normalized (0‑1)
    b = example['objects']['bbox']
    lbl= tf.cast(example['objects']['label'], tf.float32)
    ymin, xmin, ymax, xmax = tf.unstack(b, axis=1)
    boxes = tf.stack([xmin, ymin, xmax, ymax, lbl], axis=1)

    y_true = make_truth_tensor(boxes)
    return img, y_true


ds_train = (
    tfds.load('voc/2007', split='train', data_dir='~/tensorflow_datasets')
    .map(preprocess, AUTOTUNE)
    .shuffle(2056)           # or however many you want to prefetch/cache
    .cache()
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

# ──────── VALIDATION ────────
ds_val = (
    tfds.load('voc/2007', split='validation', data_dir='~/tensorflow_datasets')
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(2056)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
# # ───────────────────────────────────────────────────────── Train ─────────────
model = YoloV1_()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=yolo_v1_loss)


# def schedule(epoch):
#     if epoch < 30:
#         return 1e-4
#     elif epoch < 30:
#         return 5e-5
#     else:
#         return 1e-5


history = model.fit(ds_train,
                    validation_data=ds_val,
          epochs=100,
          steps_per_epoch=200,
          verbose=1)
print(history.history)
# import time

# start = time.time()
# imgs, y_trues = next(iter(ds_train))
# print("🕒 Data loaded:", time.time() - start)

# start = time.time()
# y_preds = model(imgs)
# print("🕒 Forward pass:", time.time() - start)

# start = time.time()
# loss = yolo_v1_loss(y_trues, y_preds)
# print("🕒 Loss computed:", time.time() - start)
