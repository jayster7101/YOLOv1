import tensorflow at tf

def cell_loss(pred_flat, ground_flat):


  pred_boxes = pred_flat[...,:10] # slies up to 10, exclusive ie 0-9
  ground_boxes = ground_flat[...,:10]

  pred_classes = pred_flat[...,10:] # slices from 10 on 10-19
  ground_classes = ground_flat[...,10:]

  

  # before computing loss, decide which img has the better IoU
  # then carry on woth loss 

  # quark of loss, seems to need to figure out the 



def yolo_loss(grid_size, pred_per_cell = 2,lambda_coord = 5,lambda_noobj = 0.5):
  def yolo_loss_compute(pred, ground):
    # input is a 7x7x30 tensor, reshape into large vector and perfrom in parallel
    # assume data is vector of 30
    # x1,y1,w1,h1,obj1, x1,y2,w2,h2,obj2, c0,c1,c2,...c19

    # reducing to two dimensions allows us to parallelize 
    pred_flat = tf.reshape(pred, [-1, 30])       # shape (49, 30) ~ infers first dim based on rest, then we make the second dim to be 30, thus it does 
                                                                  # 7*7*30 = 1490/30 from 2nd dim, giving 49 
    ground_flat = tf.reshape(ground, [-1, 30])       # shape (49, 30)






  print("test")
