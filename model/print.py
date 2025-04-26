import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]

def decode_predictions(pred, S=7, conf_thresh=0.3):
    """
    Takes raw output (7,7,30), returns list of boxes:
    [ (xmin, ymin, xmax, ymax, confidence, class_id), ... ] ~ xmin ... is wrong, its in center prediciton x,y ... ofset from the cell
    """
    boxes = []
    cell_size = 1.0 / S

    pred = pred.numpy().reshape(S, S, 30)

    for row in range(S):
        for col in range(S):
            cell = pred[row, col]
            box1 = cell[:5]
            box2 = cell[5:10]
            class_probs = cell[10:]
            print("cell", cell, "box1", box1, "box2", box2, "classprob", class_probs)

            # Pick better confidence
            if box1[4] > box2[4]:
                bx = box1
            else:
                bx = box2

            x_offset, y_offset, w_pred, h_pred, conf = bx

            if conf < conf_thresh:
                continue

            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]

            # Decode absolute center
            cx = (col + x_offset) / S
            cy = (row + y_offset) / S

            w = w_pred
            h = h_pred

            # Convert to top-left corner
            xmin = cx - w / 2
            ymin = cy - h / 2
            xmax = cx + w / 2
            ymax = cy + h / 2

            boxes.append([xmin, ymin, xmax, ymax, conf, class_id])

    return boxes

def draw_boxes_pil(image, boxes, save_path="output.png"):
    """
    image: HxWx3 NumPy image (values in [0, 1])
    boxes: list of [xmin, ymin, xmax, ymax, conf, class_id]
    """
    H, W = image.shape[:2]
    image = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.load_default()

    for box in boxes:
        xmin, ymin, xmax, ymax, conf, class_id = box
        x1 = int(xmin * W)
        y1 = int(ymin * H)
        x2 = int(xmax * W)
        y2 = int(ymax * H)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{VOC_CLASSES[class_id]} {conf:.2f}"
        draw.text((x1 + 2, y1 - 10), label, fill="white", font=font)

    pil_img.save(save_path)
    print(f"Saved annotated image to {save_path}")


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from PIL import Image

# from voc_data_gather import prep_for_inference
# VOC_CLASSES = [
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow',
#     'diningtable', 'dog', 'horse', 'motorbike', 'person',
#     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]

# def non_max_suppression(boxes, scores, iou_threshold=0.5):
#     """Apply Non-Maximum Suppression to remove overlapping boxes."""
#     if len(boxes) == 0:
#         return []
    
#     # Convert to corner format if not already
#     if boxes.shape[1] == 4:  # Already in corner format
#         boxes_corner = boxes
#     else:  # Convert from center format to corner format
#         cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#         boxes_corner = np.column_stack([
#             cx - w/2, cy - h/2,  # xmin, ymin
#             cx + w/2, cy + h/2   # xmax, ymax
#         ])
    
#     # Sort by confidence score
#     indices = np.argsort(scores)[::-1]
#     boxes_corner = boxes_corner[indices]
#     scores = scores[indices]
    
#     keep = []
#     while len(indices) > 0:
#         # Keep the box with highest confidence
#         keep.append(indices[0])
#         if len(indices) == 1:
#             break
        
#         # Calculate IoU with the rest
#         xmin1, ymin1, xmax1, ymax1 = boxes_corner[0]
#         remaining_boxes = boxes_corner[1:]
        
#         xmin2 = remaining_boxes[:, 0]
#         ymin2 = remaining_boxes[:, 1]
#         xmax2 = remaining_boxes[:, 2]
#         ymax2 = remaining_boxes[:, 3]
        
#         # Calculate intersection
#         inter_xmin = np.maximum(xmin1, xmin2)
#         inter_ymin = np.maximum(ymin1, ymin2)
#         inter_xmax = np.minimum(xmax1, xmax2)
#         inter_ymax = np.minimum(ymax1, ymax2)
        
#         inter_w = np.maximum(0, inter_xmax - inter_xmin)
#         inter_h = np.maximum(0, inter_ymax - inter_ymin)
#         inter_area = inter_w * inter_h
        
#         # Calculate union
#         box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
#         box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
#         union_area = box1_area + box2_area - inter_area
        
#         # Calculate IoU
#         iou = inter_area / (union_area + 1e-7)
        
#         # Filter boxes with IoU less than threshold
#         mask = iou <= iou_threshold
#         indices = indices[1:][mask]
#         boxes_corner = boxes_corner[1:][mask]
#         scores = scores[1:][mask]
    
#     return keep

# def visualize_yolo_predictions(model, image_path, conf_threshold=0.2, iou_threshold=0.5, grid_size=7):
#     """
#     Run YOLO inference on an image and visualize the results with proper bounding boxes.
    
#     Args:
#         model: The trained YOLO model
#         image_path: Path to the input image
#         conf_threshold: Confidence threshold for object detection
#         iou_threshold: IoU threshold for non-maximum suppression
#         grid_size: Grid size used by the model (typically 7 for YOLOv1)
#     """
#     # 1. Load and preprocess image
#     img = prep_for_inference(image_path)
#     img_display = img.numpy().copy()  # For display
    
#     # 2. Run model inference
#     img_input = tf.expand_dims(img, 0)  # Add batch dimension
#     predictions = model(img_input)[0].numpy()  # (7,7,30)
    
#     # 3. Post-process predictions
#     boxes = []
#     scores = []
#     class_ids = []
    
#     for row in range(grid_size):
#         for col in range(grid_size):
#             cell_pred = predictions[row, col]
            
#             # Get both bounding box predictions
#             box1 = cell_pred[0:5]  # x, y, w, h, conf
#             box2 = cell_pred[5:10]
            
#             # Choose box with higher confidence
#             if box1[4] >= box2[4]:
#                 box_pred, conf = box1[:4], box1[4]
#             else:
#                 box_pred, conf = box2[:4], box2[4]
            
#             # Apply activation functions
#             x = sigmoid(box_pred[0])
#             y = sigmoid(box_pred[1])
#             w = box_pred[2]**2
#             h = box_pred[3]**2
            
#             # Convert to absolute coordinates [0,1]
#             cx = (col + x) / grid_size
#             cy = (row + y) / grid_size
            
#             # Get class probabilities
#             class_probs = cell_pred[10:]
#             class_id = np.argmax(class_probs)
#             class_score = class_probs[class_id]
            
#             # Final score is confidence * class probability
#             score = conf * class_score
            
#             if score > conf_threshold:
#                 boxes.append([cx, cy, w, h])
#                 scores.append(score)
#                 class_ids.append(class_id)
    
#     # 4. Apply Non-Maximum Suppression
#     boxes = np.array(boxes)
#     scores = np.array(scores)
#     class_ids = np.array(class_ids)
    
#     if len(boxes) > 0:
#         # Group by class
#         unique_classes = np.unique(class_ids)
#         final_boxes = []
#         final_scores = []
#         final_classes = []
        
#         for cls in unique_classes:
#             cls_mask = class_ids == cls
#             cls_boxes = boxes[cls_mask]
#             cls_scores = scores[cls_mask]
            
#             keep_indices = non_max_suppression(cls_boxes, cls_scores, iou_threshold)
            
#             for idx in keep_indices:
#                 final_boxes.append(cls_boxes[idx])
#                 final_scores.append(cls_scores[idx])
#                 final_classes.append(cls)
#     else:
#         final_boxes = []
#         final_scores = []
#         final_classes = []
    
#     # 5. Visualize results
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(img_display)
    
#     for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
#         cx, cy, w, h = box
        
#         # Convert to pixel coordinates
#         H, W = img_display.shape[:2]
#         x1 = (cx - w/2) * W
#         y1 = (cy - h/2) * H
#         x2 = (cx + w/2) * W
#         y2 = (cy + h/2) * H
        
#         # Create rectangle
#         rect = Rectangle((x1, y1), x2-x1, y2-y1, 
#                          linewidth=2, edgecolor='red', facecolor='none')
#         ax.add_patch(rect)
        
#         # Add label
#         class_name = VOC_CLASSES[cls_id]
#         label = f"{class_name}: {score:.2f}"
#         ax.text(x1, y1-5, label, color='red', fontsize=10, 
#                 backgroundcolor='white')
    
#     plt.axis('off')
#     output_path = 'yolo_detection_result.png'
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close()
    
#     print(f"Detected {len(final_boxes)} objects. Visualization saved to {output_path}")
#     return final_boxes, final_scores, final_classes

# def sigmoid(x):
#     """Helper function to compute sigmoid."""
#     return 1 / (1 + np.exp(-x))

# # Usage:
# # visualize_yolo_predictions(yolov1_model, "test_img.jpg", conf_threshold=0.4)


# def save_image_properly(image, output_path="output.png"):
#     """
#     Properly save a tensor image to a file without whitewashing.
    
#     Args:
#         image: Tensor or numpy array with values in [0,1] range
#         output_path: Path to save the image
#     """
#     # Ensure we're working with numpy
#     if isinstance(image, tf.Tensor):
#         image = image.numpy()
    
#     # Ensure image is in [0,1] range
#     if np.max(image) <= 1.0:
#         # Already in [0,1], convert to [0,255]
#         image_uint8 = (image * 255).astype(np.uint8)
#     else:
#         # Assume it's already in [0,255]
#         image_uint8 = image.astype(np.uint8)
    
#     # Use PIL for saving (better quality control)
#     from PIL import Image
#     pil_image = Image.fromarray(image_uint8)
#     pil_image.save(output_path)
#     print(f"Image saved to {output_path}")

# def visualize_detections(image, boxes=None, scores=None, class_ids=None, 
#                          conf_threshold=0.2, output_path="detection_result.png"):
#     """
#     Visualize detection results with proper image quality.
    
#     Args:
#         image: Image tensor or numpy array in [0,1] range
#         boxes: List of [cx, cy, w, h] boxes in normalized coordinates
#         scores: List of confidence scores
#         class_ids: List of class IDs
#         conf_threshold: Confidence threshold
#         output_path: Path to save visualization
#     """
#     # Convert image to numpy if needed
#     if isinstance(image, tf.Tensor):
#         image = image.numpy()
    
#     # Create figure with proper DPI for quality
#     fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
#     # Display image with proper color mapping
#     ax.imshow(image)
    
#     # Draw boxes if provided
#     if boxes is not None and len(boxes) > 0:
#         for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
#             if score < conf_threshold:
#                 continue
                
#             cx, cy, w, h = box
#             x1 = (cx - w/2) * image.shape[1]
#             y1 = (cy - h/2) * image.shape[0]
#             width = w * image.shape[1]
#             height = h * image.shape[0]
            
#             # Create rectangle with thicker line for visibility
#             rect = Rectangle((x1, y1), width, height, 
#                             linewidth=3, edgecolor='red', facecolor='none')
#             ax.add_patch(rect)
            
#             # Add label with better visibility
#             class_name = VOC_CLASSES[class_id]
#             label = f"{class_name}: {score:.2f}"
#             ax.text(x1, y1-5, label, color='white', fontsize=12, 
#                     bbox=dict(facecolor='red', alpha=0.8))
    
#     # Remove axis for cleaner look
#     plt.axis('off')
    
#     # Adjust layout to eliminate whitespace
#     plt.tight_layout(pad=0)
    
#     # Save with maximum quality
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
#     plt.close(fig)
    
#     # For the absolute best quality, re-open with PIL and save again
#     try:
#         from PIL import Image
#         img = Image.open(output_path)
#         img.save(output_path, quality=95)
#     except:
#         pass
    
#     print(f"Visualization saved to {output_path}")



