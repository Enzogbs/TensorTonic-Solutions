import numpy as np

def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = np.array(box_a)
    box2_x1, box2_y1, box2_x2, box2_y2 =  np.array(box_b)

    inter_x1, inter_y1 = max(box1_x1, box2_x1), max(box1_y1, box2_y1)
    inter_x2, inter_y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)

    inter_width, inter_height = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area1, area2 =  (box1_x2 - box1_x1) * (box1_y2 - box1_y1), (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = area1 + area2 - intersection

    return intersection / union 