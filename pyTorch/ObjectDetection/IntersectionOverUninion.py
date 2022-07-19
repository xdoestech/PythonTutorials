'''
A method to determine how 'good' a predicted bounding box is
(area of overlap of predicted bbox and actual bbox)/(total area of predicted and actual bbox)
IOU should be > 0.5 (any less not accurate)
-almost perfect > 0.9
-basically impossible to get 1
'''
#Source: https://www.youtube.com/watch?v=XXYG5ZWtjj0&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=2

"""
Calculates intersection over union 

Parameters: 
    boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
    boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1, y1, x2, y2)

Returns: 
    tensor: Intersection over union for all examples
"""
import torch
#NOTE: origin at TOP RIGHT (x increases this way [>] right) (y increases this way [v] down)
#work with an arbitrary amount of data
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    #NOTE: x1,y1 > top left corner // x1,y2 bottom right corner
    #boxes_preds shape is (N,4) where N is number of bboxes
    #boxes_labels shape is (N,4)
    #NOTE: yolo algorithm will be using midpoint
    if box_format == "midpoint":
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box1_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[...,2:3] + boxes_labels[...,2:3] / 2
        box2_y2 = boxes_labels[..., 3:4] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    #NOTE: need to check for case where boxes do not overlap
        #will prevent negatives and ensure intersection is 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    #get the areas of the boxes (abs to ensure positive)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))      
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    #calculate the intersection over the union
    #NOTE: intersection subtracted from union to avoid double counting
    return intersection / (box1_area + box2_area - intersection + 1e-6)
