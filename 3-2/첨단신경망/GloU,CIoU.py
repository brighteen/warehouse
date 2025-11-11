import cv2
import numpy as np

def calculate_giou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    intersection = inter_width * inter_height
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    iou = np.round(intersection / union, 3)
    
    c_x1 = min(x1_1, x1_2)
    c_y1 = min(y1_1, y1_2)
    c_x2 = max(x2_1, x2_2)
    c_y2 = max(y2_1, y2_2)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    giou = np.round(iou - (c_area - union) / c_area, 3)
    
    return iou, giou

def calculate_ciou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    intersection = inter_width * inter_height
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    iou = intersection / union
    
    # 중심점 계산
    cx1 = (x1_1 + x2_1) / 2
    cy1 = (y1_1 + y2_1) / 2
    cx2 = (x1_2 + x2_2) / 2
    cy2 = (y1_2 + y2_2) / 2
    
    # 중심점 간 거리
    center_distance = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # 외접 박스의 대각선 거리
    c_x1 = min(x1_1, x1_2)
    c_y1 = min(y1_1, y1_2)
    c_x2 = max(x2_1, x2_2)
    c_y2 = max(y2_1, y2_2)
    diagonal_distance = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
    
    # 종횡비
    w1 = x2_1 - x1_1
    h1 = y2_1 - y1_1
    w2 = x2_2 - x1_2
    h2 = y2_2 - y1_2
    
    v = (4 / (np.pi ** 2)) * ((np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2)
    alpha = v / (1 - iou + v)
    ciou = iou - (center_distance / diagonal_distance) - alpha * v
    
    return np.round(ciou, 3)


img = cv2.imread('cutecat.jpg')

ground_truth = [73, 43, 588, 791]
predicted_box = [50, 20, 400, 500]

# IoU & GIoU & CIoU
iou, giou = calculate_giou(ground_truth, predicted_box)
ciou = calculate_ciou(ground_truth, predicted_box)

print('IoU:', iou)
print('GIoU:', giou)
print('CIoU:', ciou)

cv2.rectangle(img, (ground_truth[0], ground_truth[1]), 
              (ground_truth[2], ground_truth[3]), (0, 255, 0)) # 초록색
cv2.rectangle(img, (predicted_box[0], predicted_box[1]), 
              (predicted_box[2], predicted_box[3]), (0, 0, 255)) # 빨간색

cv2.imwrite('result.jpg', img)