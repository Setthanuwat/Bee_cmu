import mrcnn
import mrcnn.config
import mrcnn.model
import cv2
import os
import matplotlib.pyplot as plt
import random
import colorsys
import numpy as np

# กำหนด class label
CLASS_NAMES = ['BG', 'Bee']

# กำหนด config ของ Mask R-CNN
class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)

# สร้างโมเดล Mask R-CNN สำหรับการ inference
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# โหลด weights ของโมเดลที่เทรนไว้
model.load_weights(filepath="Bee_mask_rcnn_trained7.h5", 
                   by_name=True)

# โหลดรูปภาพและแปลงจาก BGR เป็น RGB
image = cv2.imread("Bee.v9i.voc/test/FA3-2_jpeg.rf.3ed2a1bef58b9bddf44a85902d0e93a8.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ใช้โมเดลในการตรวจจับวัตถุ
r = model.detect([image], verbose=0)
r = r[0]

# ฟังก์ชันกำหนดสีสำหรับแต่ละคลาส
def get_class_color(class_id, class_names):
    if class_names[class_id] == 'Bee':
        return (1.0, 1.0, 0.0)  # สีเหลือง (RGB ใน matplotlib คือค่าในช่วง 0-1)
    else:
        return (1.0, 1.0, 1.0)  # สีขาวสำหรับ background หรือคลาสอื่น

# ฟังก์ชันแสดง mask ด้วยสีที่กำหนดเอง
def display_instances(image, boxes, masks, class_ids, class_names, scores=None, min_score=0.2):
    N = boxes.shape[0]
    fig, ax = plt.subplots(1, figsize=(12, 12))
    
    # แสดงรูปภาพ
    ax.imshow(image)

    # สำหรับแต่ละวัตถุ
    for i in range(N):
        score = scores[i] if scores is not None else None
        
        # เช็คคะแนนว่าต่ำกว่าค่าที่กำหนดหรือไม่
        if score is not None and score < min_score:
            continue  # ข้ามไปยังวัตถุต่อไปถ้าคะแนนต่ำกว่าค่าที่กำหนด
        
        class_id = class_ids[i]
        mask = masks[:, :, i]
        
        # กำหนดสีของ mask
        color = get_class_color(class_id, class_names)
        
        # Apply mask ด้วยสีที่กำหนดเอง
        masked_image = image.copy()
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1,
                                             masked_image[:, :, c] * (1 - 0.5) + 0.5 * color[c] * 255,
                                             masked_image[:, :, c])

        ax.imshow(masked_image)

        # วาด bounding box
        y1, x1, y2, x2 = boxes[i]
        p = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                          alpha=0.7, linestyle="dashed",
                          edgecolor=color)
        ax.add_patch(p)

        # แสดง class label และคะแนน
        label = class_names[class_id]
        caption = f"{label} {score:.3f}" if score else label
        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

    plt.axis('off')
    plt.show()


# แสดงผลลัพธ์โดยใช้สีที่กำหนดเอง
display_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'])
