import tensorflow as tf
import tensorflow_datasets as tfds
import math
from voc_data_gather import get_Data, resize_and_rescale, prep_for_inference
from yolo_loss import make_truth_tensor
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from yolov1 import YoloV1_
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from yolo_loss import yolo_loss,center_to_topLeftBottomRight

from utils import image_show_w_boxes, decode_and_draw

import numpy as np
import os
from PIL import Image
from print import decode_predictions, draw_boxes_pil



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def decode_yolo_output(pred, S=7, conf_thresh=0.3):
    boxes = []
    pred = pred.numpy().reshape(S, S, 30)

    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            b1 = cell[:5]
            b2 = cell[5:10]
            class_logits = cell[10:]

            # Activation: box 1
            x1 = tf.sigmoid(b1[0]).numpy()
            y1 = tf.sigmoid(b1[1]).numpy()
            w1 = tf.square(b1[2]).numpy()
            h1 = tf.square(b1[3]).numpy()
            c1 = tf.sigmoid(b1[4]).numpy()

            # Activation: box 2
            x2 = tf.sigmoid(b2[0]).numpy()
            y2 = tf.sigmoid(b2[1]).numpy()
            w2 = tf.square(b2[2]).numpy()
            h2 = tf.square(b2[3]).numpy()
            c2 = tf.sigmoid(b2[4]).numpy()

            # Pick better confidence box
            if c1 > c2:
                x, y, w, h, conf = x1, y1, w1, h1, c1
            else:
                x, y, w, h, conf = x2, y2, w2, h2, c2

            # Center → Absolute coords
            cx = (j + x) / S
            cy = (i + y) / S
            xmin = cx - w / 2
            ymin = cy - h / 2
            xmax = cx + w / 2
            ymax = cy + h / 2

            # Softmax on class logits → get P(class | object)
            class_probs = tf.nn.softmax(class_logits).numpy()
            class_id = np.argmax(class_probs)
            class_prob = class_probs[class_id]

            # Final score = P(object) * P(class | object)
            score = conf * class_prob

            # Filter out bad predictions
            if (
                score < conf_thresh or
                w <= 0 or h <= 0 or
                xmin >= xmax or ymin >= ymax or
                not (0 <= xmin <= 1 and 0 <= ymin <= 1 and xmax <= 1 and ymax <= 1)
            ):
                continue

            boxes.append([xmin, ymin, xmax, ymax, score, class_id])

    return boxes



VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]

def visualize_and_save(model=None,
                       data_dir='~/tensorflow_datasets',
                       target_size=448,
                       grid_size=7):
    # 1) grab one preprocessed sample
    ds = (
      tfds.load('voc/2007', split='train', data_dir=data_dir)
        .map(preprocess, tf.data.AUTOTUNE)
    )
    image, y_true = next(iter(ds.take(1)))
    img_np = image.numpy()

    # 2) common figure setup
    def _make_fig():
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img_np)
        ax.axis('off')
        return fig, ax

    # 3) draw GT
    fig, ax = _make_fig()
    for i in range(grid_size):
      for j in range(grid_size):
        cell = y_true[i,j].numpy()
        if np.any(cell[:4] > 0):
          xmin, ymin, xmax, ymax = cell[:4]
          cls = np.argmax(cell[4:])
          # to pixels:
          x1, y1 = xmin*target_size, ymin*target_size
          w, h    = (xmax-xmin)*target_size, (ymax-ymin)*target_size
          rect = Rectangle((x1,y1), w, h,
                           linewidth=2, edgecolor='green', facecolor='none')
          ax.add_patch(rect)
          ax.text(x1, y1-2, VOC_CLASSES[cls],
                  color='green', fontsize=8, backgroundcolor='white')
    fig.savefig('gt_sample.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 4) if you gave me a model, overlay preds
    if model is not None:
      pred = model(tf.expand_dims(image,0))[0].numpy()  # (7,7,30)
      flat = pred.reshape(-1,30)
      fig, ax = _make_fig()
      for idx, out in enumerate(flat):
        # pick higher-conf box
        if out[4] >= out[9]:
          x_raw,y_raw,w_raw,h_raw,conf = out[:5]
        else:
          x_raw,y_raw,w_raw,h_raw,conf = out[5:10]
        if conf < 0.001:
          continue
        r, c = divmod(idx, grid_size)
        cx = (c + x_raw)/grid_size
        cy = (r + y_raw)/grid_size
        w_abs, h_abs = w_raw, h_raw
        xmin, ymin = cx - w_abs/2, cy - h_abs/2
        xmax, ymax = cx + w_abs/2, cy + h_abs/2
        # to pixels
        x1, y1 = xmin*target_size, ymin*target_size
        wpx, hpx = (xmax-xmin)*target_size, (ymax-ymin)*target_size
        cls = np.argmax(out[10:])
        rect = Rectangle((x1,y1), wpx, hpx,
                         linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-2, VOC_CLASSES[cls],
                color='red', fontsize=8, backgroundcolor='white')
      fig.savefig('pred_sample.png', bbox_inches='tight', pad_inches=0)
      plt.close(fig)

    print("→ Saved gt_sample.png", 
          "and pred_sample.png (if model was provided)")

# Usage:
# visualize_and_save(model=yolov1)




def save_weights_fp16(model, path="yolov1_fp16_weights_yoloLR.npz"):
    weights = model.get_weights()
    weights_fp16 = [w.astype(np.float16) for w in weights]
    np.savez_compressed(path, *weights_fp16)
    print(f"Saved FP16 weights to {path}")


# tf.config.optimizer.set_jit(True) 

AUTOTUNE   = tf.data.AUTOTUNE

def preprocess(sample):
    image, raw_targets = resize_and_rescale(sample)  # returns (448,448,3), and a (N,5) [xmin,ymin,xmax,ymax,cls]
    y_true = make_truth_tensor(raw_targets, grid_size=7, num_classes=20)  # → (49, 24)
    y_true = tf.reshape(y_true, (7, 7, 24))

    # **shape asserts** to catch mistakes early**
    tf.debugging.assert_equal(tf.shape(image), [448, 448, 3],
                              message="Image must be (448,448,3)")
    tf.debugging.assert_equal(tf.shape(y_true), [7, 7, 24],
                              message="y_true must be (7,7,24)")

    return image, y_true


# ds_raw = tfds.load('voc/2007', split='train', data_dir='/cs/student/jaydenjardine/tensorflow_datasets')


# apply preprocess → batch → prefetch

# VOC 2007 train set size ≈ 5011
TOTAL_SAMPLES = 5011
BATCH_SIZE = 5
STEPS_PER_EPOCH = math.ceil(TOTAL_SAMPLES / BATCH_SIZE)  # = 314
class BatchLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        loss = logs["loss"]
        if loss > 10.0:   # or whatever threshold
            tf.print("🔥 Batch", batch, "loss spiked to", loss)




def overfit():
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 1
    STEPS_PER_EPOCH = 5
    # Load only 5 training examples
    train_dataset = (
        tfds.load('voc/2007', split='train[:5]', data_dir='~/tensorflow_datasets')
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    img, y_true = next(iter(train_dataset))
    y = y_true[0].numpy()  # shape (7,7,24)
    for i in range(7):
        for j in range(7):
            cell = y[i,j]
            if cell[:4].any(): 
                print(f"Cell [{i},{j}]:")
                print(f"  Box coordinates: {cell[:4]}")
                print(f"  Detailed breakdown:")
                print(f"    xmin: {cell[0]}")
                print(f"    ymin: {cell[1]}")
                print(f"    xmax: {cell[2]}")
                print(f"    ymax: {cell[3]}")
                print(f"  Class-onehot sum: {cell[4:].sum()}")

    # Use same 5 for evaluation
    val_dataset = (
        tfds.load('voc/2007', split='train[:5]', data_dir='~/tensorflow_datasets')
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # 2) Exponential LR decay schedule
    initial_lr = 1e-5
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=initial_lr,
    #     decay_steps=1000,    # e.g. ~200 steps per epoch × 5 epochs 
    #     decay_rate=0.95,
    #     staircase=True
    # )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=100,
        decay_rate=0.9
    )
    # 3) Optimizer with decay + gradient clipping
    opt = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    # Build model
    yolov1 = YoloV1_(input_size=448)
    yolov1.compile(optimizer= opt, loss=yolo_loss(7))

    # Train on small data
    history = yolov1.fit(
        train_dataset,
        epochs=300,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_dataset,
        validation_steps=5,
        # callbacks=[lr_callback]
    )

    # Save weights
    os.makedirs("checkpoints", exist_ok=True)
    save_weights_fp16(yolov1,"overfit.npz")

    # Test on same 5 examples and visualize
    for i, (img, _) in enumerate(val_dataset.take(5)):
        pred = yolov1(img)[0]
        boxes = decode_yolo_output(pred, S=7, conf_thresh=0.05)
        img_np = img[0].numpy()
        draw_boxes_pil(img_np, boxes, save_path=f"overfit_out_{i}.png")
        print(f"Saved output for sample {i}.")


def main():
    dataset = (
        tfds.load('voc/2007', split='train', data_dir='~/tensorflow_datasets')
        .map(preprocess, AUTOTUNE)
        .cache()
        .shuffle(5011)
        .repeat()
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    for d in dataset.take(1):
        print(d)
        la = d[1]

        img0 = la[0]  # shape: (7, 7, 24)

        # Loop over all cells and print
        for row in range(7):
            for col in range(7):
                cell = img0[row, col]  # shape: (24,)
                print(f"Cell[{row},{col}]:", cell.numpy())


    def safe_yolo_lr_schedule(epoch): # this one is better for ADAM
        if epoch < 5:
            return 1e-5 + (epoch / 5) * (3e-4 - 1e-5)  # Warmup
        elif epoch < 50:
            return 3e-4
        elif epoch < 75:
            return 1e-4
        else:
            return 3e-5

    lr_callback = tf.keras.callbacks.LearningRateScheduler(safe_yolo_lr_schedule)


    # initial_lr = 3e-5
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=initial_lr,
    #     decay_steps=1000,        # Try 1000 or STEPS_PER_EPOCH
    #     decay_rate=0.95,         # Decays to 95% every 1000 steps
    #     staircase=True           # Use step-wise decay
    # )

    # opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    # # callbacks = [
    # #     BatchLogger(),
    # # ]
    eval_dataset = (
    tfds.load('voc/2007', split='validation', data_dir='~/tensorflow_datasets')
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


    yolov1 = YoloV1_(input_size=448)
    yolov1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=yolo_loss(7))
    history = yolov1.fit(dataset.repeat(),
          epochs=15,
          validation_data = eval_dataset,
          steps_per_epoch=STEPS_PER_EPOCH,
          callbacks=[lr_callback]
    )



    img_s = prep_for_inference("test_img.jpg")
    Image.fromarray((img_s.numpy() * 255).astype(np.uint8)).save("check_input.png")
    img = prep_for_inference("test_img.jpg")  # (448, 448, 3), float32 [0,1]
    print(" - shape:", img.shape)
    print(" - min/max:", np.min(img), np.max(img))

    img4d = tf.expand_dims(img, 0)            # (1,448,448,3)

    pred = yolov1(img4d)[0]  # shape (7, 7, 30)


    boxes = decode_yolo_output(pred, S=7, conf_thresh=0.01)
    draw_boxes_pil(img_s.numpy(), boxes, save_path="inference_output.png")
        #   callbacks=callbacks)

    # yolov1.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
    #           loss=yolo_loss(grid_size=7))

    # history = yolov1.fit(dataset,
    #         epochs=15,
    #         verbose=1)


    os.makedirs("checkpoints", exist_ok=True)
    save_weights_fp16(yolov1)

    # yolov1.save_weights("checkpoints/yolov1_ckpt.keras")



    # print(history.history)

    # # 2) prep your image
    # img = prep_for_inference("test_img.jpg")     # [H,W,3] float32 [0,1]
    # img4d = tf.expand_dims(img, 0)               # (1,H,W,3)

    # # 3) forward + reshape
    # pred = yolov1(img4d)[0]                       # (7,7,30)
    
    # flat = tf.reshape(pred, (-1, 30))            # (49,30)

    # # 4) decode & draw *all* boxes (threshold=0 to debug)
    # decode_and_draw(img, flat, conf_thresh=0.85, S=7)

    # test_img = prep_for_inference("test_img.jpg")
    # test_img = tf.expand_dims(test_img, axis=0)  # (1,H,W,3)
    # pred = yolov1(test_img)[0]                   # (7,7,30)
    # flat = tf.reshape(pred, (-1,30))             # (49,30)

    # decode_and_draw(test_img[0], flat)


def load_weights_fp16(model, path="yolov1_fp16_weights_3.npz"):
    data = np.load(path)
    weights_fp32 = [data[f'arr_{i}'].astype(np.float32) for i in range(len(data.files))]
    model.set_weights(weights_fp32)
    print(f"Loaded weights from {path}")
    return model

def test():
    img_s = prep_for_inference("cat.jpg")
    Image.fromarray((img_s.numpy() * 255).astype(np.uint8)).save("check_input.png")
    # 1) build and load
    model = YoloV1_(input_size=448)
    model = load_weights_fp16(model)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            print(f"Dropout rate: {layer.rate}, trainable: {layer.trainable}")


        # 1. Run model inference
    img = prep_for_inference("cat.jpg")  # (448, 448, 3), float32 [0,1]
    print(" - shape:", img.shape)
    print(" - min/max:", np.min(img), np.max(img))

    img4d = tf.expand_dims(img, 0)            # (1,448,448,3)

    pred = model(img4d,training=False)[0]  # shape (7, 7, 30)


    boxes = decode_yolo_output(pred, S=7, conf_thresh=0.01)
    draw_boxes_pil(img_s.numpy(), boxes, save_path="inference_output.png")

    # # 2. Decode predictions
    # boxes = decode_predictions(pred, S=7, conf_thresh=0.5)

    # # 3. Draw with PIL
    # draw_boxes_pil(img.numpy(), boxes, save_path="yolo_output.png")

    # visualize_yolo_predictions(model, "cat.jpg", conf_threshold=0.4)
    # # visualize_and_save(model=model)

    # # 2) prep your image
    # img = prep_for_inference("cat.jpg")     # [H,W,3] float32 [0,1]
    # img4d = tf.expand_dims(img, 0)               # (1,H,W,3)

    # # 3) forward + reshape
    # pred = model(img4d)[0]                       # (7,7,30)
    # flat = tf.reshape(pred, (-1, 30))            # (49,30)
    # for p in flat:
    #    print("Pred",p)

    # # 4) decode & draw *all* boxes (threshold=0 to debug)
    # decode_and_draw(img, flat, conf_thresh=0.0, S=7)

# test()
# main()
overfit()




# test_img = prep_for_inference("test_img.jpg")
# test_img = tf.expand_dims(test_img, axis=0)
# if test_img is not None:
#     infer = yolov1(test_img)
#     # print("model output", infer)
#     flat = tf.reshape(infer[0], (-1, 30))

#     for i, out in enumerate(flat):
#         if max(out[4],out[9]) > 0.001: # choose better confidence 
#             if(out[4] == max(out[4],out[9])):
#                 x,y,w,h = out[0:4]
#             else:
#                 x,y,w,h = out[5:9]

#             box = [x,y,w,h]
#             image_show_w_boxes(test_img[0],i,box) # MUST FIX, when printing, you must pass in the cell as well, since center point is RELATIVE to cell!!!!!
#         print(out)
#         input("SHow")








    # image_show_w_boxes(test_img, bbox)




# pred = yolov1()

# history = yolov1.fit(
#     dataset,
#     epochs=1,
#     steps_per_epoch=100,
#     verbose=1
# )
# print(history.history)

# # take one batch
# imgs, y_trues = next(iter(dataset))
# loss_fn = yolo_loss(7)
# # forward pass
# y_preds = yolov1(imgs)  
# print("y_preds shape:", y_preds.shape)  
# print("y_trues", y_trues)
# # → should be (16, 7, 7, 30)

# # compute loss on that batch
# per_image_losses = loss_fn(y_trues, y_preds)


# print("batch loss:", per_image_losses)
# dataset = tfds.load('voc/2007', split='train', data_dir='/cs/student/jaydenjardine/tensorflow_datasets')
# # inspect shapes
# for ds in dataset.take(1):
#     print("ds", ds)
#     # print(imgs.shape)     # (16, 448, 448, 3)
#     # print(y_trues.shape)  # (16,   7,   7, 24)

# # dataset = (ds_raw
#            .map(preprocess, num_parallel_calls=AUTOTUNE)
#            .batch(BATCH_SIZE, drop_remainder=True)
#            .prefetch(AUTOTUNE))

# for example in ds_raw.take(1):
#     print("modifed")
#     print(example)
#     resized = resize_and_rescale(example)
#     print("resized", resized)



# # Collect dataset and scale accordingly 
# ds  = get_Data()
# ds = tfds.load('voc/2007',  split='train', data_dir='/cs/student/jaydenjardine/tensorflow_datasets')
# print("ds", ds)
# for example in ds.take(1):
#     print("example",example)

# dataset = ds.map(resize_and_rescale, num_parallel_calls=tf.data.AUTOTUNE)
# dataset = dataset.map(preprocess)
# dataset = dataset.map(lambda x, y: (tf.expand_dims(x, 0), tf.expand_dims(y, 0)))

# for example in dataset.take(1):
#     print("modifed")
#     print(example)


# # dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)



# yolov1.summary()

# for imgs, y_trues in dataset.take(1):
#     print("→ imgs:", imgs.shape)       # (BATCH_SIZE, 448,448,3)
#     print("→ y_trues:", y_trues.shape) # (BATCH_SIZE,   7,  7,24)

# # loss_fn = yolo_loss(7)
# # yolov1.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss=loss_fn
# )

# # BATCH_SIZE = 16
# # dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# dataset = dataset.prefetch(tf.data.AUTOTUNE)


# for image, labels in dataset.take(1):
#     y_pred = yolov1(image)
#     loss_val = loss_fn(y_pred, labels)
#     print("Loss:", loss_val.numpy())


# # Train it
# history = yolov1.fit(
#     dataset,
#     epochs=50,
#     steps_per_epoch=100,   # You can set this based on dataset size
#     verbose=1
# )




