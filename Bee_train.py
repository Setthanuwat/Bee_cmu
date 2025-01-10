import os
import xml.etree.ElementTree as ET
from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model

import imgaug.augmenters as iaa

class BeeDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        self.add_class("dataset", 1, "Bee")
        subset_dir = os.path.join(dataset_dir, subset)
        for filename in os.listdir(subset_dir):
            if filename.endswith('.jpg'):
                image_id = filename[:-4]
                img_path = os.path.join(subset_dir, filename)
                ann_path = os.path.join(subset_dir, image_id + '.xml')
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('Bee'))
        return masks, asarray(class_ids, dtype='int32')

    def extract_boxes(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class BeeConfig(mrcnn.config.Config):
    NAME = "Bee_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # ลดลงจาก 2 เป็น 1 เพื่อประหยัดหน่วยความจำ
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 15 # จะปรับค่านี้ในภายหลัง
    DETECTION_MIN_CONFIDENCE = 0.8
    # LEARNING_RATE = 0.001

# โหลดและเตรียมข้อมูล
train_dataset = BeeDataset()
train_dataset.load_dataset(dataset_dir='Bee.v9i.voc', subset='train')
train_dataset.prepare()

validation_dataset = BeeDataset()
validation_dataset.load_dataset(dataset_dir='Bee.v9i.voc', subset='valid')
validation_dataset.prepare()

# กำหนดค่า configuration
bee_config = BeeConfig()
bee_config.STEPS_PER_EPOCH = len(train_dataset.image_ids) // (bee_config.IMAGES_PER_GPU * bee_config.GPU_COUNT)

# สร้างโมเดล
model = mrcnn.model.MaskRCNN(mode='training', model_dir='./', config=bee_config)

# โหลด pre-trained weights
model.load_weights(filepath='Bee_mask_rcnn_trained6.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])




# กำหนด data augmentation
# augmentation = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     iaa.Affine(rotate=(-45, 45)),
#     iaa.Multiply((0.8, 1.2))
# ])

# ฝึกโมเดล
model.train(train_dataset=train_dataset,
            val_dataset=validation_dataset,
            learning_rate=bee_config.LEARNING_RATE,
            epochs=20,
            layers='heads',)

# บันทึกโมเดลที่ฝึกแล้ว
model_path = 'Bee_mask_rcnn_trained7.h5'
model.keras_model.save_weights(model_path)
