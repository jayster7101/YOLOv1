import tensorflow as tf
import numpy as np
EPS = 1e-7 # term to help with numerical stability an not have model shit the bed over small nums or possibly div by  zero 
# ground_flat format is  xmin, ymin, xmax,ymax
# pred_flat will be in x_center, y_center, w, h all normalised 

def center_to_topLeftBottomRight(bbox_pred):
  x, y, w, h = tf.unstack(bbox_pred, num=4, axis=-1)

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
  xmin2, ymin2, xmax2, ymax2 = tf.unstack(ground[:4], num=4, axis=-1)

  # print("IOU",  xmin1, ymin1, xmax1, ymax1)

  # Intersection box
  inter_xmin = tf.maximum(xmin1, xmin2) # finds largest of the left side of max for x coord
  inter_ymin = tf.maximum(ymin1, ymin2)
  inter_xmax = tf.minimum(xmax1, xmax2)
  inter_ymax = tf.minimum(ymax1, ymax2)

  # Intersection area, code accounts for the case that boxes dont overlap thus implying the area is 0 
  inter_w = tf.maximum(inter_xmax - inter_xmin, 0)
  inter_h = tf.maximum(inter_ymax - inter_ymin, 0)
  inter_area = inter_w * inter_h

    # Areas of each box
  area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
  area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

  # Union area
  union_area = area1 + area2 - inter_area + EPS

  return inter_area / union_area 

def pixel_to_cell(x,y,image_size = 448, grid_size = 7):
  cell_size = image_size / grid_size # our case cell_size will be 64
  cell_x = int(x*image_size//cell_size) # temp un-normalize for easy compute
  cell_y = int(y*image_size//cell_size)
  cell = int(cell_y* grid_size + cell_x)
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

def make_truth_tensor(truth, grid_size, num_classes=20):
    """
    truth input: [batch, ]
    truth: tf.Tensor of shape (N, 5) where each row is [x_min, y_min, x_max, y_max, class_id]
    Returns: Tensor of shape (S*S, 4+num_classes)
    """
    total_cells = grid_size * grid_size

    cell_size = 448/grid_size
    # N = tf.shape(truth)[0]

    # Get box centers
    xmin, ymin, xmax, ymax = tf.split(truth[:, :4], 4, axis=1)
    x_center = (xmin + xmax) / 2 #! We have not fixed to offset bbox
    y_center = (ymin + ymax) / 2

    # Compute cell index of linearized truth cell 0:48
    cell_x = tf.cast(x_center * grid_size, tf.int32)
    cell_y = tf.cast(y_center * grid_size, tf.int32)
    cell_idx = cell_y * grid_size + cell_x  # shape (N, 1)

    # location of topleft relative to entire image
    top_left_x = cell_x / grid_size 
    top_left_y = cell_y / grid_size

    # x_center_realtive_to_cell = tf.abs(top_left_x - x_center)
    # y_center_realtive_to_cell = tf.abs(top_left_y - y_center)



    # One-hot class vector
    class_ids = tf.cast(truth[:, 4], tf.int32)
    class_vecs = tf.one_hot(class_ids, depth=num_classes, dtype=tf.float32) # one hot encoding of the class vector 

    # print("xmin shape:", xmin.shape)
    # print("class_vecs shape:", class_vecs.shape)


    # Full representation: [bbox, class_vec]
    full_vec = tf.concat([xmin, ymin, xmax, ymax, class_vecs], axis=1)

    # Tensor scatter update
    gt_array = tf.zeros((total_cells, 4 + num_classes), dtype=tf.float32) # create the linearized 
    gt_array = tf.tensor_scatter_nd_update(
        gt_array,
        cell_idx,     # now shape (N,1)
        full_vec      # shape (N,4+num_classes) ~ 49,24
    )


    return gt_array 


# # Ground will be a vector of length s * s,
# # at pos[i], will return bbox and label or 
# def cell_loss(pred_flat, ground, lambda_coord = 5.0, lambda_noobj = 0.5):
  
#     has_obj = tf.reduce_any(ground[:4] != 0.0 ) # checks to see if the ground truth predicts anything

#     pred_iou_1 = IoU(pred_flat[:4],ground[0:4]) # if obj has non zero IoU
#     pred_iou_2 = IoU(pred_flat[5:9],ground[0:4])

#     best_IoU = tf.maximum(pred_iou_1, pred_iou_2) # chooses the better of the two
#     best_pred = tf.where(pred_iou_1 >= pred_iou_2, # assigns the better iod pred 
#                          pred_flat[0:5],pred_flat[5:10])
                         
    
#     def obj_branch():

#         # Localization Loss (coordinate prediction)        
#         x, y, w, h, conf = tf.unstack(best_pred[:5], num=5, axis=-1)   # unpack values
#         gt_x, gt_y, gt_w, gt_h = tl_to_center(tf.unstack(ground[:4], num=4, axis=-1) ) # get the center of the ground truths 

#         coord_loss = tf.square(x - gt_x) + tf.square(y - gt_y) # find loss due to x and y

#         # print("Coord loss from x and y ", coord_loss)
        

#         coord_loss += tf.square(tf.sqrt(tf.abs(w) +  1e-6)) - tf.sqrt(tf.abs(gt_w)+ 1e-6) # find loss due to w & h
#         coord_loss += tf.square(tf.sqrt(tf.abs(h) + 1e-6)) - tf.sqrt(tf.abs(gt_h) + 1e-6)

#         # Confidence loss
#         obj_loss = tf.square(conf - best_IoU) # confidence - best IoU is the obj loss

#         # Classification loss
#         # print("pred_flat" , pred_flat, "ground", ground)
#         class_loss = tf.reduce_sum(tf.square(pred_flat[10:] - ground[4:])) # since its a loop, you can parallelize 

#         total_loss = lambda_coord * coord_loss + obj_loss + class_loss

#         return total_loss
    
#     def noobj_branch():
#        return lambda_noobj * (pred_flat[4]**2 + pred_flat[9]**2)
    
#     return tf.cond(has_obj, obj_branch, noobj_branch)
      

# # def yolo_loss(grid_size, pred_per_cell = 2,lambda_coord = 5,lambda_noobj = 0.5, num_of_classes = 20):
# #   # Ground will be an array of length s * s so that we can allow for quick parallel checking 

# #   def yolo_loss_compute(pred, truth): # pred is of shape (batch,7,7,30), truth is of (batch, 7,7,24) could refactror an takeout as done twice now 
# #     batch_size  = tf.shape(pred)[0]
# #     pred_flat  = tf.reshape(pred, (batch_size, -1, pred_per_cell*5 + num_of_classes)) # shape (batch, SxS, 30)
# #     truth_flat = tf.reshape(truth, (batch_size, -1, 4 + num_of_classes ))  # shape (batch, SxS, 24)

# #     # map cell_loss over each (pred,T)
# #     # returns shape (B, S*S)
# #     cell_losses = tf.map_fn(
# #         lambda x: cell_loss(x[0], x[1], lambda_coord, lambda_noobj),
# #         (pred_flat, truth_flat),
# #         dtype=tf.float32
# #     )

# #     loss_per_image = tf.reduce_sum(cell_losses, axis=1)  # (B,) sum across batch
# #     return tf.reduce_mean(loss_per_image)   # now we avg loss over batch 
# #   return yolo_loss_compute


# def yolo_loss(grid_size,
#               pred_per_cell=2,
#               lambda_coord=5.0,
#               lambda_noobj=0.5,
#               num_classes=20):
#     S  = grid_size
#     B5 = pred_per_cell*5 + num_classes
#     C4 = 4 + num_classes

#     def loss_fn(y_true, y_pred):
#         # y_pred: (B,S,S,B5),  y_true: (B,S,S,C4)
#         B = tf.shape(y_pred)[0]

#         # flatten into (B*S*S, feat)
#         P_all = tf.reshape(y_pred, (-1, B5))   # (B*49,30)
#         T_all = tf.reshape(y_true, (-1, C4))   # (B*49,24)

#         # compute one loss per cell
#         losses_all = tf.map_fn(
#             lambda pair: cell_loss(pair[0], pair[1],
#                                    lambda_coord, lambda_noobj),
#             (P_all, T_all),
#             dtype=tf.float32
#         )  # → (B*49,)

#         # reshape back into (B,49) and sum per image
#         losses_per_image = tf.reshape(losses_all, (B, S*S))
#         per_image_sum   = tf.reduce_sum(losses_per_image, axis=1)  # (B,)

#         return tf.reduce_mean(per_image_sum)  # scalar

#     return loss_fn


##############


EPS = 1e-7  # just one EPS everywhere

# def cell_loss(pred_flat, ground, idx, S, lambda_coord=5.0, lambda_noobj=0.5):
    
#      # unpack raw
#     x1r, y1r, w1r, h1r, c1r = tf.unstack(pred_flat[0:5])
#     x2r, y2r, w2r, h2r, c2r = tf.unstack(pred_flat[5:10])

#     # — apply activations —
#     x1 = tf.sigmoid(x1r);  y1 = tf.sigmoid(y1r)
#     w1 = w1r;   h1 = h1r
#     conf1 = tf.sigmoid(c1r)

#     x2 = tf.sigmoid(x2r);  y2 = tf.sigmoid(y2r)
#     w2 = w2r;   h2 = h2r
#     conf2 = tf.sigmoid(c2r)

#     # convert to absolute [0,1] coords
#     row = idx // S;  col = idx % S
#     cx1 = (tf.cast(col,tf.float32) + x1)/S
#     cy1 = (tf.cast(row,tf.float32) + y1)/S
#     cx2 = (tf.cast(col,tf.float32) + x2)/S
#     cy2 = (tf.cast(row,tf.float32) + y2)/S

#     box1 = tf.stack([cx1, cy1, w1, h1, conf1], axis=-1)
#     box2 = tf.stack([cx2, cy2, w2, h2, conf2], axis=-1)

#     # now your IoUs will stay in [0,1] and not blow up
#     iou1 = IoU(box1[:4], ground[:4])
#     iou2 = IoU(box2[:4], ground[:4])
#     best_IoU = tf.maximum(iou1, iou2)
#     best_pred = tf.where(iou1 >= iou2, box1, box2)

#     # row = idx // S
#     # col = idx %  S
#     # # 1) is there an object in this cell?
#     has_obj = tf.reduce_any(ground[:4] != 0.0)

#     # x_cell, y_cell, w, h, conf = tf.unstack(pred_flat[0:5]) # we need to restack back 
#     # x_abs = (tf.cast(col,tf.float32) + x_cell) / S
#     # y_abs = (tf.cast(row,tf.float32) + y_cell) / S
#     # pred_1 = tf.stack([x_abs,y_abs,w,h,conf], axis=-1)

#     # x_cell2, y_cell2, w2, h2, conf2 = tf.unstack(pred_flat[5:10]) # we need to restack back
#     # x_abs2 = (tf.cast(col,tf.float32) + x_cell2) / S
#     # y_abs2 = (tf.cast(row,tf.float32) + y_cell2) / S
#     # pred_2= tf.stack([x_abs2,y_abs2,w2,h2,conf2], axis=-1)



#     # # 2) compute both IoUs
#     # iou1 = IoU(pred_1[0:4], ground[0:4])
#     # iou2 = IoU(pred_2[0:4], ground[0:4])
#     # best_IoU_raw = tf.maximum(iou1, iou2)

#     # # 3) pick the “best” prediction vector
#     # best_pred = tf.where(iou1 >= iou2,
#     #                      pred_1,  # x,y,w,h,conf
#     #                      pred_2)

#     def obj_branch():
#         # unpack raw
#         x, y, w_raw, h_raw, conf_raw = tf.unstack(best_pred, num=5)

#         # ⭑ force positivity
#         w = tf.maximum(w_raw, EPS)
#         h = tf.maximum(h_raw, EPS)

#         # GT center + size
#         gt_x, gt_y, gt_w, gt_h = tl_to_center(tf.unstack(ground[:4], num=4))

#         # ⭑ coord‐xy loss
#         coord_xy = tf.square(x - gt_x) + tf.square(y - gt_y)

#         # ⭑ coord‐wh loss (√w – √gt_w)² + (√h – √gt_h)²
#         coord_wh = (
#             tf.square(tf.sqrt(w + EPS)  - tf.sqrt(gt_w + EPS)) +
#             tf.square(tf.sqrt(h + EPS)  - tf.sqrt(gt_h + EPS))
#         )

#         coord_loss = coord_xy + coord_wh

#         # ⭑ confidence loss, both clamped to [0,1]
#         conf    = tf.clip_by_value(conf_raw,    0.0, 1.0)
#         bestIoU = tf.clip_by_value(best_IoU, 0.0, 1.0)
#         obj_loss = tf.square(conf - bestIoU)

#         # classification loss (unchanged)
#         class_loss = tf.reduce_sum(tf.square(pred_flat[10:] - ground[4:]))

#         return lambda_coord * coord_loss + obj_loss + class_loss

#     def noobj_branch():
#         # penalize both conf heads when no object
#         return lambda_noobj * (
#             tf.square(pred_flat[4]) + tf.square(pred_flat[9])
#         )

#     return tf.cond(has_obj, obj_branch, noobj_branch)

def cell_loss(pred_flat, ground, idx, S, lambda_coord=5.0, lambda_noobj=0.5):
    EPS = 1e-7

    # Unpack prediction ~ realitive to top left of cell 
    x1r, y1r, w1r, h1r, c1r = tf.unstack(pred_flat[0:5])
    x2r, y2r, w2r, h2r, c2r = tf.unstack(pred_flat[5:10])

    # Apply YOLO activations
    x1 = tf.sigmoid(x1r)
    y1 = tf.sigmoid(y1r)
    w1 = tf.maximum(w1r, EPS)  # assumes w is already sqrt(w)
    h1 = tf.maximum(h1r, EPS)
    conf1 = tf.sigmoid(c1r)

    x2 = tf.sigmoid(x2r)
    y2 = tf.sigmoid(y2r)
    w2 = tf.maximum(w2r, EPS)
    h2 = tf.maximum(h2r, EPS)
    conf2 = tf.sigmoid(c2r)

    # Convert cell offset to absolute coords
    row = idx // S
    col = idx % S

    # Now cx<#> is realtive to whole image 
    cx1 = (tf.cast(col, tf.float32) + x1) / S
    cy1 = (tf.cast(row, tf.float32) + y1) / S
    cx2 = (tf.cast(col, tf.float32) + x2) / S
    cy2 = (tf.cast(row, tf.float32) + y2) / S

    box1_iou = tf.stack([cx1, cy1, w1, h1, conf1])
    box_1_cell = tf.stack([x1, y1, w1, h1, conf1])

    box2_iou = tf.stack([cx2, cy2, w2, h2, conf1])
    box_2_cell = tf.stack([x2, y2, w2, h2, conf2])

    # Ground truth: [xmin, ymin, xmax, ymax] ~ these are still reltive to the whole image, vs they sh
    xmin, ymin, xmax, ymax = tf.unstack(ground[:4])
    gt_w = xmax - xmin
    gt_h = ymax - ymin
    gt_cx = xmin + gt_w / 2 # need to make realtive to cell top left of idx
    gt_cy = ymin + gt_h / 2

    row = tf.math.floordiv(idx, S)
    col = tf.math.floormod(idx, S)

    cell_x = tf.cast(col, tf.float32) / S
    cell_y = tf.cast(row, tf.float32) / S

    x_offset = (gt_cx - cell_x) 
    y_offset = (gt_cy - cell_y)
    

    gt_box = tf.stack([gt_cx, gt_cy, gt_w, gt_h])

    # IoU
    iou1 = IoU(box1_iou, ground[:4])
    iou2 = IoU(box2_iou, ground[:4])
    best_iou = tf.maximum(iou1, iou2)
    best_iou = tf.clip_by_value(best_iou, 0.05, 1.0) # clip to not allow crazy IoUs

    best_box = tf.where(iou1 >= iou2, box_1_cell, box_2_cell) # this returns the best cell, but returns cell values rel to cell top left 
    best_conf = best_box[4]
    best_conf = tf.clip_by_value(best_conf, 0.0, 1.0) # clip conf as we dont want wildly pos or neg conf

    # Compute loss if there's an object
    has_obj = tf.reduce_any(ground[:4] != 0.0)
    def obj_branch():
      pred_cx, pred_cy, pred_w, pred_h = tf.unstack(best_box[:4]) 

        # Compute individual component losses
      x_center_loss = tf.square(pred_cx - x_offset)
      y_center_loss = tf.square(pred_cy - y_offset)
      

      coord_loss = (
         x_center_loss + y_center_loss + 

            tf.square(pred_cx - x_offset) + 
            tf.square(pred_cy - y_offset) 
            +
            tf.square( (tf.math.sign(pred_w) * tf.sqrt(tf.abs(pred_w)) + EPS) - tf.sqrt(gt_w + EPS)) +
            tf.square( (tf.math.sign(pred_h) * tf.sqrt(tf.abs(pred_h) + EPS)) - tf.sqrt(gt_h + EPS))
        )
      
      # # Use tf.print for debugging during graph execution
      # tf.print("Coordinate Transformation Debugging:")
      # tf.print("Cell coordinates - col:", col, "row:", row)
      # tf.print("Cell top-left x:", cell_x, "y:", cell_y)
      # tf.print("Ground truth center - cx:", gt_cx, "cy:", gt_cy)
      # tf.print("Predicted center - cx:", pred_cx, "cy:", pred_cy)
      # tf.print("x_offset:", x_offset, "y_offset:", y_offset)
      # tf.print("Predicted width:", pred_w, "Ground truth width:", gt_w)
      # tf.print("Predicted height:", pred_h, "Ground truth height:", gt_h)
      
      # Coordinate loss focusing on relative coordinates
        
    #   # Compute individual loss components with more insight
    #   x_center_loss = tf.square(pred_cx - x_offset)
    #   y_center_loss = tf.square(pred_cy - y_offset)
    #   width_loss = tf.square(pred_w - gt_w)
    #   height_loss = tf.square(pred_h - gt_h)
      
    #   # Collect and potentially log detailed information
    #   coord_loss_components = {
    #       'x_center_loss': x_center_loss,
    #       'y_center_loss': y_center_loss,
    #       'width_loss': width_loss,
    #       'height_loss': height_loss
    #   }
    
    # # Compute total coordinate loss
    #   coord_loss = sum(coord_loss_components.values())
      
      # tf.print("Coordinate Loss:", coord_loss)
      


    # def obj_branch():
        # pred_cx, pred_cy, pred_w, pred_h = tf.unstack(best_box[:4]) 
        # coord_loss = (
        #     tf.square(pred_cx - x_offset) + # Now should be realtive to top left of cell they are in  
        #     tf.square(pred_cy - y_offset) +
        #     tf.square( (tf.math.sign(pred_w) * tf.sqrt(tf.abs(pred_w)) + EPS) - tf.sqrt(gt_w + EPS)) +
        #     tf.square( (tf.math.sign(pred_h) * tf.sqrt(tf.abs(pred_h) + EPS)) - tf.sqrt(gt_h + EPS))
        # )

      obj_loss = tf.square((best_conf * best_iou) -1 ) # This is what that paper stated, can later clamp to never allow IoU less than some value 
        # obj_loss = tf.square(best_conf -1 )

            # Sigmoid cross-entropy loss
      class_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels= ground[4:], 
            logits=pred_flat[10:]
        )
      class_loss =  tf.reduce_mean(class_loss)
        # class_loss = tf.reduce_sum(tf.square(pred_flat[10:] - ground[4:]))

      return lambda_coord * coord_loss + obj_loss + class_loss
        # # return class_loss

    def noobj_branch():
        return lambda_noobj * (tf.square(conf1) + tf.square(conf2))



    return tf.cond(has_obj, obj_branch, noobj_branch)



def yolo_loss(grid_size,
              pred_per_cell=2,
              lambda_coord=5.0,
              lambda_noobj=0.5,
              num_classes=20):
    S  = grid_size
    B5 = pred_per_cell*5 + num_classes   # = 30
    C4 = 4 + num_classes                  # = 24

    def loss_fn(y_true, y_pred):
        # flatten (batch*49, feat)
        B = tf.shape(y_pred)[0]

        # build [0,1,2,...,48] for one image, then tile for B images → shape (B*49,)
        cell_idxs = tf.range(S*S, dtype=tf.int32)           # (49,)
        cell_idxs = tf.tile(cell_idxs[None, :], [B, 1])       # (B,49)
        cell_idxs = tf.reshape(cell_idxs, [-1])               # (B*49,)

        P_all = tf.reshape(y_pred, (-1, B5))
        T_all = tf.reshape(y_true, (-1, C4))

        # map over every cell
        losses_all = tf.map_fn(
            lambda pair: cell_loss(pair[0], pair[1], pair[2], S ,
                                   lambda_coord, lambda_noobj),
            (P_all, T_all, cell_idxs),
            dtype=tf.float32
        )  # → (B*49,)

        # sum per‐image and mean
        losses_per_image = tf.reshape(losses_all, (B, S*S))
        per_image_sum   = tf.reduce_sum(losses_per_image, axis=1)

        return tf.reduce_mean(per_image_sum)

    return loss_fn
