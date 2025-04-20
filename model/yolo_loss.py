import tensorflow as tf
import numpy as np

# ground_flat format is  xmin, ymin, xmax,ymax
# pred_flat will be in x_center, y_center, w, h all normalised 

def center_to_topLeftBottomRight(bbox_pred):
  x, y, w, h = bbox_pred

  xmin = x - w/2
  xmax = x + w/2
  ymin = y - h/2
  ymax = y + h/2
  
  return xmin, ymin, xmax, ymax

# input to be passed in: bbox
# Format: xmin, ymin, xmax, ymax
# Output: x_center, y_center, width, height 
def tl_to_center(bbox):
  xmin, ymin, xmax, ymax = bbox

  width = xmax - xmin
  x_center = xmin + width/2

  height = ymax - ymin

  y_center = ymin + height/2

  return x_center, y_center, width, height





# pred is x_center, y_center, width, height 
# ground is xmin,ymin, xmax,ymax ~ topleft, bottom right
def IoU(pred, ground):
  xmin1, ymin1, xmax1, ymax1 = center_to_topLeftBottomRight(pred[:4]) # put the pred into same format at truth
  xmin2, ymin2, xmax2, ymax2 = ground[:4] # this is in normalised form 

  # Intersection box
  inter_xmin = max(xmin1, xmin2) # finds largest of the left side of max for x coord
  inter_ymin = max(ymin1, ymin2)
  inter_xmax = min(xmax1, xmax2)
  inter_ymax = min(ymax1, ymax2)

  # Intersection area, code accounts for the case that boxes dont overlap thus implying the area is 0 
  inter_w = max(inter_xmax - inter_xmin, 0)
  inter_h = max(inter_ymax - inter_ymin, 0)
  inter_area = inter_w * inter_h

    # Areas of each box
  area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
  area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

  # Union area
  union_area = area1 + area2 - inter_area

  return inter_area / union_area

def pixel_to_cell(x,y,image_size = 448, grid_size = 7):
  cell_size = image_size/ grid_size # our case cell_size will be 64
  cell_x = int(x*image_size//cell_size) # temp un-normalize for easy compute
  cell_y = int(y*image_size//cell_size)
  cell = int(cell_y*grid_size + cell_x)
  return cell


'''
 Assumes:
 - All data is Pre-normalized, all data is pre scaled to 448x448 size
 - in centered format

 Known issue:
  - since in normalized form, it could be the case that values very close to 1 may round up
  - if issue seem to cause prob: possible fix cell_x = min(grid_size - 1, int(x * image_size // cell_size))
'''
def inCell(pred_center, truth_center, image_size = 448, grid_size = 7):
  x,y = pred_center
  cell = pixel_to_cell(x,y,image_size=image_size,grid_size=grid_size)

  x_T,y_T = truth_center
  cellT = pixel_to_cell(x_T,y_T,image_size=image_size,grid_size=grid_size)

  return int(cellT == cell)


# Ground will be a vector of length s * s,
# at pos[i], will return bbox and label or 
def cell_loss(pred_flat, ground, lambda_coord = 5.0, lambda_noobj = 0.5):

  cell = pixel_to_cell(pred_flat[0],pred_flat[1]) # now we know the cell
  print("Cell", cell)
  print(ground)

  if tf.reduce_any(ground[:4] != 0): # if truth in cell 
    print("pred box matches truth")
      # Unpack ground truth
    gt_box = ground[0:4]  # assumed in [x_min, y_min, x_max, y_max]
    gt_class = ground[4:]  # one-hot class vector
    class_pred = pred_flat[10:]  # shape (20,)
    # Find best IoU pred
    pred_iou_1 = IoU(pred_flat[:4],gt_box) # if obj has non zero IoU
    pred_iou_2 = IoU(pred_flat[5:9],gt_box)
    best_pred = pred_flat[:5] if pred_iou_1 >= pred_iou_2 else pred_flat[5:10] # now will hold the x_center, y_center, width, height, confidence for best pred box 
    best_IoU = tf.maximum(pred_iou_1,pred_iou_2)

    # Localization Loss (coordinate prediction)
    x, y, w, h, conf = best_pred
    gt_x, gt_y, gt_w, gt_h = tl_to_center(gt_box)

    coord_loss = tf.square(x - gt_x) + tf.square(y - gt_y)

    print("Coord loss from x and y ", coord_loss)
    

    coord_loss += tf.square(tf.sqrt(tf.maximum(tf.abs(w), 1e-6)) - tf.sqrt(tf.abs(gt_w)))
    coord_loss += tf.square(tf.sqrt(tf.maximum(tf.abs(h), 1e-6)) - tf.sqrt(tf.abs(gt_h)))

    # Confidence loss
    obj_loss = tf.square(conf - best_IoU) # confidence - best IoU is the obj loss

    # Classification loss
    class_loss = tf.reduce_sum(tf.square(class_pred - gt_class)) # since its a loop, you can parallelize 

    total_loss = lambda_coord * coord_loss + obj_loss + class_loss
    noobj_loss = tf.constant(0.0)
  else:
    coord_loss = tf.constant(0.0)
    obj_loss = tf.constant(0.0)
    class_loss = tf.constant(0.0)
    print("pred_flat[4&9]",pred_flat[4],pred_flat[9])
    noobj_loss = tf.square(pred_flat[4]) + tf.square(pred_flat[9])
    total_loss = lambda_noobj * noobj_loss

  print("Coord loss:", coord_loss.numpy())
  print("Obj loss:", obj_loss.numpy())
  print("Class loss:", class_loss.numpy())
  print("Noobj loss:", noobj_loss.numpy())


  return total_loss

    
 # if ground Truth is designated for this iteration of the cell, then we do this block of code, else the other
  #  now compare this for each of the ground truths to see 
  # if obj center point of GT not even in the cell, we apply the noobj

  # get list of truth cells, add to map, if map contains cells, 1 else 0 


  # pred_boxes = pred_flat[...,:10] # slies up to 10, exclusive ie 0-9
  # xmin1, ymin1, xmax1, ymax1 = center_to_topLeftBottomRight(pred_flat[:4]) # in top left format
  # pred_class_1_conf = pred_flat[4] # confidence 
  # xmin2, ymin2, xmax2, ymax2 = center_to_topLeftBottomRight(pred_flat[5:9])
  # pred_class_2_conf = pred_flat[9] # confidence 



  # # intersection = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)


  # ground_boxes = ground_flat[...,:10]

  # pred_classes = pred_flat[...,10:] # slices from 10 on 10-19
  # ground_classes = ground_flat[...,10:]

  # areab1 = (pred_boxes[2] - pred_boxes[0] )* (pred_boxes[3] - pred_boxes[1]) # xmin, ymin, xmax,ymax
  # areab2 = (pred_boxes[6] - pred_boxes[4] )* (pred_boxes[7] - pred_boxes[5]) # xmin, ymin, xmax,ymax

  # # intersection = 


 # find which box to use first, then well convert to yolo x_center, y_center, etc 
  # before computing loss, decide which img has the better IoU
  # then carry on woth loss 

  # quark of loss, seems to need to figure out the 

def make_truth_tensor(truth, grid_size, num_classes=20):
    """
    truth: tf.Tensor of shape (N, 5) where each row is [x_min, y_min, x_max, y_max, class_id]
    Returns: Tensor of shape (S*S, 4+num_classes)
    """
    total_cells = grid_size * grid_size
    # N = tf.shape(truth)[0]

    # Get box centers
    xmin, ymin, xmax, ymax = tf.split(truth[:, :4], 4, axis=1)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    # Compute cell index
    cell_x = tf.cast(x_center * grid_size, tf.int32)
    cell_y = tf.cast(y_center * grid_size, tf.int32)
    cell_idx = cell_y * grid_size + cell_x  # shape (N, 1)

    # One-hot class vector
    class_ids = tf.cast(truth[:, 4], tf.int32)
    class_vecs = tf.one_hot(class_ids, depth=num_classes, dtype=tf.float32)

    # Full representation: [bbox, class_vec]
    full_vec = tf.concat([truth[:, :4], class_vecs], axis=1)

    # Tensor scatter update
    gt_array = tf.zeros((total_cells, 4 + num_classes), dtype=tf.float32)
    gt_array = tf.tensor_scatter_nd_update(
        gt_array,
        cell_idx,     # now shape (N,1)
        full_vec      # shape (N,4+num_classes)
    )


    return gt_array

def yolo_loss(grid_size, pred_per_cell = 2,lambda_coord = 5,lambda_noobj = 0.5, num_of_classes = 20):
  # Ground will be an array of length s * s so that we can allow for quick parallel checking 

  def yolo_loss_compute(pred, truth):
    # print(pred)
    pred_flat = tf.reshape(pred, [-1, 30])
    # print(pred_flat)
    truth_vec = make_truth_tensor(truth, grid_size)

    losses = []
    for i in range(grid_size*grid_size):
        losses.append(cell_loss(pred_flat[i],
                                truth_vec[i],
                                lambda_coord,
                                lambda_noobj))
    # tf.add_n sums a list of tensors elementwise
    return tf.add_n(losses)
    # # input is a 7x7x30 tensor, reshape into large vector and perfrom in parallel
    # # assume data is vector of 30
    # # x1,y1,w1,h1,obj1, x1,y2,w2,h2,obj2, c0,c1,c2,...c19

    # # reducing to two dimensions allows us to parallelize 
    # pred_flat  = tf.reshape(pred, [-1, 5*pred_per_cell + num_of_classes])  # (S*S,30)
    # truth_vec  = make_truth_tensor(truth, grid_size, num_of_classes)       # (S*S,4+C)
    #    # shape (49, 30) ~ infers first dim based on rest, then we make the second dim to be 30, thus it does 
    #                                                               # 7*7*30 = 1490/30 from 2nd dim, giving 49 
    # # truth_vec = make_truth_tensor(truth=truth, grid_size=grid_size)

    # cell_loss = tf.map_fn(
    #         lambda pair: cell_loss(pair[0], pair[1], lambda_coord, lambda_noobj),
    #         (pred_flat, truth_vec),
    #         fn_output_signature=tf.float32
    #     )

    # total_loss = tf.reduce_sum(cell_loss)
    # return total_loss

  return yolo_loss_compute




    # total_cells = grid_size * grid_size


    # gt_array = [tf.zeros(4 + 24, dtype=tf.float32) for _ in range(total_cells)]

    # for ground_item in ground:
    #   bbox = ground_item[0:4]
    #   cell = pixel_to_cell(bbox)
    #   class_id = ground_item[4]
    #   class_vec = tf.one_hot(class_id,depth=num_of_classes)
    #   gt_array[cell] = tf.concat([bbox, class_vec], axis=0)
      


    # ground_flat = tf.reshape(ground, [-1, 30])       # this wont even be of length, it will actually be a list of xmin,ymin,xmax,ymax,class

    # can reshape the tensor passed so that we encode a pred to have data else it woull be zero and its confidence and predict instead 
    # this allows us to check to see quickly, if the bbox are zero we jsut 
    #  now call cell func on whole array 




# loss_fn = yolo_loss(grid_size=7)
# # dummy_pred  = tf.zeros((7,7,5*2+20))

# # Create (7*7=49) cells × 30
# pred_flat = np.zeros((49, 30), dtype=np.float32)

# # Fill in cell index where object exists
# cell_idx = pixel_to_cell(0.28, 0.4, image_size=1, grid_size=7)
# pred_flat[cell_idx] = [
#     0.28, 0.4, 0.6, 0.7, .76,    # box 1
#     0.28, 0.2, 0.36, 0.7, .76,   # box 2
#     0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # class 4 one-hot (starts at index 10)
#     0, 0, 0, 0, 0,0, 0, 0, 0, 0
# ]

# dummy_pred = tf.constant(pred_flat.reshape(7, 7, 30), dtype=tf.float32)
# # dummy_pred = tf.constant([[0.28, 0.4, 0.6, 0.7, .76, 0.28, 0.2, 0.36, 0.7, .76,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=tf.float32)
# dummy_truth = tf.constant([[0.3, 0.4, 0.6, 0.7, 2],[0.3, 0.4, 0.6, 0.7, 2]], dtype=tf.float32) # class type is 0 based
# # dummy_truth = tf.zeros((0,5))  # no objects
# print(loss_fn(dummy_pred, dummy_truth).numpy())  # should run without errors

# Define loss function
loss_fn = yolo_loss(grid_size=7)

# # Create prediction tensor (7x7 grid, 30 values per cell)
# pred_flat = np.zeros((49, 30), dtype=np.float32)

# # Place prediction in correct cell
# cell_idx = pixel_to_cell(0.28, 0.4, image_size=1, grid_size=7)
# print("cell_idx", cell_idx)
# pred_flat[cell_idx] = [
#     0.28, 0.4, 0.6, 0.7, .5,    # box 1 (centered + width/height + conf)
#     0.28, 0.2, 0.36, 0.7, .76,   # box 2
#     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  # class 1 one-hot (index 11 = class 1)
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# ]

# # # Reshape back into YOLO prediction shape (7, 7, 30)
# # dummy_pred = tf.constant(pred_flat.reshape(7, 7, 30), dtype=tf.float32)

# # bbox_pred = [0.28, 0.4, 0.6, 0.7]
# # x_min, y_min, x_max, y_max = center_to_topLeftBottomRight(bbox_pred)

# # # Ground truth box: [x_min, y_min, x_max, y_max, class_id]
# # dummy_truth = tf.constant([[x_min, y_min, x_max, y_max, 1],[x_min, y_min, x_max, y_max, 2]], dtype=tf.float32)

# # # Compute loss
# # print("YOLO Loss:", loss_fn(dummy_pred, dummy_truth).numpy())