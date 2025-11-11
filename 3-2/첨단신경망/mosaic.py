import cv2
import numpy as np
import matplotlib.pyplot as plt

def mosaic(images, boxes, labels, output_size=(640, 640)):

    h, w = output_size
    xc = np.random.randint(int(w * 0.3), int(w * 0.7))
    yc = np.random.randint(int(h * 0.3), int(h * 0.7))
    
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_boxes = []
    mosaic_labels = []
    
    # 4개 영역 정의
    positions = [
        (0, 0, xc, yc),      # 좌상단
        (xc, 0, w, yc),      # 우상단
        (0, yc, xc, h),      # 좌하단
        (xc, yc, w, h)       # 우하단
    ]
    
    for (x1, y1, x2, y2), img, img_boxes, img_labels in zip(
            positions, images, boxes, labels):
        
        # 이미지 크기
        target_w = x2 - x1
        target_h = y2 - y1
        
        # 원본 이미지 크기
        img_h, img_w = img.shape[:2]
        
        # 스케일 계산
        scale_x = target_w / img_w
        scale_y = target_h / img_h
        
        # 이미지 리사이즈 및 배치
        resized = cv2.resize(img, (target_w, target_h))
        mosaic_img[y1:y2, x1:x2] = resized
        
        # Bounding Box 변환
        for box, label in zip(img_boxes, img_labels):
            box_x, box_y, box_w, box_h = box
            
            # 새로운 좌표 계산
            new_x = box_x * scale_x + x1
            new_y = box_y * scale_y + y1
            new_w = box_w * scale_x
            new_h = box_h * scale_y
            
            mosaic_boxes.append([new_x, new_y, new_w, new_h])
            mosaic_labels.append(label)

    return mosaic_img, np.array(mosaic_boxes), np.array(mosaic_labels)


def draw_boxes(image, boxes):
    img_draw = image.copy()
    color = (0, 255, 0)  # 초록색
    
    # 각 박스 그리기
    for box in boxes:
        x, y, w, h = box
        
        # 정수 변환
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
    
    return img_draw


image_files = ['cutecat.jpg', 'cutedog.jpg', 'cutebird.jpg', 'cuterabbit.jpg']
images = []

for path in image_files:
    img = cv2.imread(path)
    images.append(img)

boxes = [  
    [[73, 43, 588, 791]],  # img1 object boxes
    [[120, 50, 320, 310]], # img2 object boxes
    [[0, 60, 325, 480]],   # img3 object boxes
    [[44, 64, 384, 717]]   # img4 object boxes
]

labels = [[0],[1],[2],[3]] # 각 이미지 클래스 레이블 예시

mosaic_img, mosaic_boxes, mosaic_labels = mosaic(images, boxes, labels)

print('Mosaic image shape:', mosaic_img.shape)
print('Boxes:', '\n' , mosaic_boxes)
print('mosaic_labels:', mosaic_labels)

mosaic_with_boxes = draw_boxes(mosaic_img, mosaic_boxes)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.imshow(cv2.cvtColor(mosaic_with_boxes, cv2.COLOR_BGR2RGB))
plt.savefig('mosaic.jpg')
plt.show()