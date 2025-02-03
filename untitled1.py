#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:34:46 2018

@author: avsthiago
"""

import numpy as np
import cv2
import os
# เปลี่ยนการนำเข้าโมดูล
from tensorflow.keras.models import load_model

from keras.applications.imagenet_utils import preprocess_input
import math
from tqdm import tqdm
from collections import Counter
import datetime
import warnings
import imghdr
from pathlib import PurePath

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PATH = os.path.dirname(os.path.realpath("_file_"))

PATH_IMAGES = os.path.join(*list(PurePath("../original_images/").parts))
PATH_MODEL = "model"
PATH_DETECTIONS = os.path.join(*list(PurePath("../annotations/detections/").parts))
PATH_PREDICTIONS = os.path.join(*list(PurePath("../annotations/predictions/").parts))
PATH_OUT_IMAGE = os.path.join(*list(PurePath("../output/labeled_images/").parts))
PATH_OUT_CSV = os.path.join(*list(PurePath("../output/spreadsheet/").parts))
MIN_CONFIDENCE = 0.9995

LEFT_BAR_SIZE = 4
img_size = 64
batch_size = 16



def cross_plataform_directory():
    global PATH_IMAGES, PATH_DETECTIONS, PATH_PREDICTIONS, PATH_OUT_IMAGE, PATH_OUT_CSV
    if "\\" in PATH_IMAGES:
        PATH_IMAGES += "\\"
        PATH_DETECTIONS += "\\"
        PATH_PREDICTIONS += "\\"
        PATH_OUT_IMAGE += "\\"
        PATH_OUT_CSV += "\\"
    elif "/" in PATH_IMAGES:
        PATH_IMAGES += "/"
        PATH_DETECTIONS += "/"
        PATH_PREDICTIONS += "/"
        PATH_OUT_IMAGE += "/"
        PATH_OUT_CSV += "/"


def get_qtd_by_class(points, labels):
    points_filtered = points[points[:, 4] == 1, 3]
    sum_predictions = Counter(points_filtered)
    return [
        *[str(sum_predictions[i]) for i, j in enumerate(labels)],
        str(len(points_filtered)),
    ]


def get_header(labels):
    return "Img Name," + ",".join([i for i in labels]) + ",Total\n"


def draw_labels_bar(image, labels, colors):
    height = image.shape[0]
    left_panel = np.zeros((height, LEFT_BAR_SIZE, 3), dtype=np.uint8)
    labels = [l.title() for l in labels]

    for i, cl in enumerate(zip(colors, labels)):
        color, label = cl
        cv2.putText(
            left_panel,
            " ".join([str(i + 1), ".", label]),
            (15, 70 * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            color,
            2,
        )

    return np.hstack((left_panel, image))


def draw_circles_labels(image, labels, points, colors=None, draw_labels=True):
    """ปรับฟังก์ชันวาดวงกลมให้มีขนาดเล็กลง"""
    if colors is None:
        colors = [
            (255, 0, 0),
            (0, 255, 255),
            (0, 0, 128),
            (255, 0, 255),
            (0, 255, 0),
            (255, 255, 100),
            (0, 0, 255),
        ]

    if draw_labels:
        image = draw_labels_bar(np.copy(image), labels, colors)

    points[:, 0] += LEFT_BAR_SIZE

    # ปรับความหนาของเส้นวงกลมให้บางลง
    for p in points:
        cv2.circle(image, (p[0], p[1]), p[2], colors[p[3]], 1)  # ปรับความหนาเส้นเป็น 1

    points[:, 0] -= LEFT_BAR_SIZE
    return image

def extract_circles(
    image, pts, output_size=64, mean_radius_default=32, standardize_radius=True
):
    """
    Extract circular regions of interest (ROIs) from an image.
    
    Parameters
    ----------
    image : ndarray
        Input image with full size.
    pts : ndarray
        Array of points in the shape [N, 3] where each row is [x, y, r].
        x, y are the center coordinates and r is the radius.
    output_size : int, optional
        The desired output size (width and height) of each ROI (default is 224x224).
    mean_radius_default : int, optional
        The base size used to standardize all circle radii (default is 32).
    standardize_radius : bool, optional
        If True, radii will be scaled relative to mean_radius_default.
    
    Returns
    -------
    ROIs : list of ndarray
        List of cropped and resized ROIs as images.
    """
    ROIs = []
    
    try:
        # Copy points array to avoid modifying original
        points = np.copy(pts)

        if standardize_radius:
            # Adjust radii to standard size
            points[:, 2] = output_size / mean_radius_default * points[:, 2]

        # Determine the required border size based on the largest radius
        max_radius = int(points[:, 2].max() + 1)
        
        # Add a border to the image to avoid out-of-bound errors
        img_with_border = cv2.copyMakeBorder(
            image,
            max_radius, max_radius, max_radius, max_radius,
            cv2.BORDER_REFLECT
        )

        # Adjust the center coordinates to account for the added border
        points[:, [0, 1]] += max_radius

        # Loop through each point to crop and resize
        for point in points:
            x, y, r = int(point[0]), int(point[1]), int(point[2])
            
            # Calculate crop boundaries
            x1, y1 = x - r, y - r  # Top-left corner
            x2, y2 = x + r, y + r  # Bottom-right corner
            
            # Safely crop the region
            cropped = img_with_border[y1:y2, x1:x2]
            
            # Check if the cropped region is valid
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                # Resize to the desired output size
                resized = cv2.resize(cropped, (output_size, output_size))
                ROIs.append(resized)
        
    except Exception as e:
        print(f"Error in extract_circles: {str(e)}")

    return ROIs


def classify_image(im_name, npy_name, labels, net, img_size, file):
    try:
        if not os.path.isfile(im_name):
            raise
        if not os.path.isfile(npy_name):
            raise

        image = cv2.imread(im_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points = np.load(npy_name)

        pt = np.copy(points)
        pt[:, 2] = pt[:, 2] // 2

        blob_imgs = extract_circles(image, np.copy(pt), output_size=img_size)
        blob_imgs = np.asarray([i for i in blob_imgs])
        blob_imgs = preprocess_input(blob_imgs)

        scores = None

        for chunk in [
            blob_imgs[x : x + batch_size] for x in range(0, len(blob_imgs), batch_size)
        ]:
            output = net.predict(chunk)

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))

        lb_predictions = np.argmax(scores, axis=1)
        vals_predictions = np.amax(scores, axis=1)

        points_pred = np.hstack(
            (np.copy(points), np.expand_dims(lb_predictions, axis=0).T)
        )

        sum_predictions = Counter(lb_predictions)
        lb = [j + " " + str(sum_predictions[i]) for i, j in enumerate(labels)]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_predita = draw_circles_labels(image, lb, points_pred)

        inside_roi = np.ones_like(points_pred[:, 3])
        new_class = np.copy(points_pred[:, 3])

        st_use_retrain = (vals_predictions > MIN_CONFIDENCE) * 1

        csl = np.vstack(
            [i for i in [new_class, st_use_retrain, inside_roi, vals_predictions]]
        ).T

        points_pred = np.hstack((points_pred, csl))

        if file is not None:
            file.write(
                ",".join(
                    [im_name.split("/")[-1], *get_qtd_by_class(points_pred, labels)]
                )
                + "\n"
            )

        date_saved = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
        height, width, _ = image.shape
        roi = ((0, 0), (width, height))

        array_to_save = np.array([roi, date_saved, points_pred])

        if PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]:
            dest_folder = os.path.join(
                PATH_PREDICTIONS,
                os.path.join(*PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]),
            )
        else:
            dest_folder = PATH_PREDICTIONS

        array_name = PurePath(im_name).parts[-1].split(".")[:-1][0] + ".npy"
        array_name = os.path.join(dest_folder, array_name)

        create_folder(array_name)
        np.save(array_name, array_to_save)

        out_img_name = os.path.join(PATH_OUT_IMAGE, im_name.replace(PATH_IMAGES, ""))
        create_folder(out_img_name)
        cv2.imwrite(out_img_name, cv2.resize(img_predita, (1280, 720)))
    except Exception as e:
        print("\nFiled to classify image " + im_name, e)


def segmentation(img, model):
    """ปรับฟังก์ชัน segmentation สำหรับภาพขนาด 1280x720"""
    IMG_WIDTH_DEST = 128
    IMG_HEIGHT_DEST = 128
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    original_shape = img.shape[:2]

    if original_shape != (720, 1280):
        img = cv2.resize(img, (1280, 720))

    reflect = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_REFLECT)

    pos_x = np.arange(0, 1248, 128)
    pos_y = np.arange(0, 688, 128)
    slices = [
        np.s_[y[0]:y[1], x[0]:x[1]]
        for x in zip(pos_x, pos_x + 160)
        for y in zip(pos_y, pos_y + 160)
    ]

    X = np.zeros((len(slices), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    for j, sl in enumerate(slices):
        X[j] = cv2.resize(reflect[sl], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)

    preds = model.predict(X)
    preds = (preds > 0.5).astype(np.uint8)

    RESULT_Y = np.zeros((len(preds), IMG_HEIGHT_DEST, IMG_WIDTH_DEST, 1), dtype=np.uint8)

    for j, x in enumerate(preds):
        RESULT_Y[j] = np.expand_dims(
            cv2.resize(x, (160, 160), interpolation=cv2.INTER_LINEAR)[16:144, 16:144],
            axis=-1,
        )

    reconstructed_mask = np.squeeze(np.hstack([np.vstack(i) for i in np.split(RESULT_Y, len(pos_y))]))[
        32:752, 32:1312
    ] * 255

    if original_shape != (720, 1280):
        reconstructed_mask = cv2.resize(reconstructed_mask, (original_shape[1], original_shape[0]))

      # remove internal areas
    _, contours, _ = cv2.findContours(reconstructed_mask, 1, 2)
    max_cnt = contours[np.argmax(np.array([cv2.contourArea(i) for i in contours]))]

    reconstructed_mask *= 0
    cv2.drawContours(reconstructed_mask, [max_cnt], 0, (255, 255, 255), -1)

    bounding_rect = cv2.boundingRect(max_cnt)  # x,y,w,h

    return reconstructed_mask, bounding_rect


import cv2
import numpy as np
import os
from pathlib import PurePath
def find_circles(im_name, img, mask, cnt, output_size=64):
    """
    Find small circles in the image with adjusted parameters and debug visualization.
    """
    try:
        x, y, w, h = cnt
        x1, y1 = 0, 0   
        x2, y2 = 1280, 720   

        # Crop the image to the defined ROI
        cropped_image = img[y1:y2, x1:x2]

        # Convert to grayscale
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # แสดงภาพ grayscale เพื่อ debug
       # cv2.imshow("Grayscale", gray)
      #  cv2.waitKey(1000)

        # ปรับ parameters ของ GaussianBlur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # ลดขนาด kernel และปรับ sigma
        
        # แสดงภาพที่ blur แล้วเพื่อ debug
        #cv2.imshow("Blurred", blurred)
       # cv2.waitKey(1000)

        # ปรับ parameters ของ HoughCircles ให้ผ่อนคลายกว่าเดิม
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=4.7,  # ลดจาก 10 เป็น 1
            minDist=7,  # เพิ่มระยะห่างขั้นต่ำระหว่างวงกลม
            param1=100,  # ค่า gradient ขั้นต่ำ
            param2=30,  # threshold ของ accumulator
            minRadius=4,  # เพิ่มขนาดขั้นต่ำ
            maxRadius=10  # เพิ่มขนาดสูงสุด
        )

        # If circles are detected, process them
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Store cropped circles and circle data
            cropped_circles = []
            circle_data = []
            
            # สร้างภาพ copy สำหรับวาด debug
            debug_image = cropped_image.copy()
            
            for (cx, cy, r) in circles:
                # Ensure the circle is within bounds
                if cx - r > 0 and cy - r > 0 and cx + r < cropped_image.shape[1] and cy + r < cropped_image.shape[0]:
                    # Store circle coordinates and radius
                    circle_data.append([cx, cy, r])
                    
                    # วาดวงกลมที่ตรวจพบ
                    cv2.circle(debug_image, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(debug_image, (cx, cy), 2, (0, 0, 255), 3)
                    
                    # Crop the circle
                    try:
                        cropped_circle = cropped_image[cy - r: cy + r, cx - r: cx + r]
                        if cropped_circle.size > 0:  # ตรวจสอบว่าได้ภาพจริงๆ
                            resized_circle = cv2.resize(cropped_circle, (output_size, output_size))
                            cropped_circles.append(resized_circle)
                    except Exception as e:
                        print(f"Error cropping circle: {str(e)}")
                        continue

            # แสดงภาพที่วาดวงกลมแล้ว
            cv2.imshow("Detected Circles", debug_image)
            cv2.waitKey(1000)

            # Save circle data as NPY file
            if circle_data:
                npy_filename = os.path.splitext(im_name)[0] + '.npy'
                output_path = os.path.join(PATH_DETECTIONS, npy_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                circle_data = np.array(circle_data)
                np.save(output_path, circle_data)
                print(f"Saved {len(circle_data)} circles to {output_path}")
            else:
                print("No circles detected")

            cv2.destroyAllWindows()
            return cropped_circles
        else:
            print("No circles detected by HoughCircles")
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Cell detection failed on image {PurePath(im_name).parts[-1]}: {str(e)}")
        cv2.destroyAllWindows()

    return []

def create_folder(path):
    path = os.path.join(*PurePath(path).parts[:-1])
    if not os.path.exists(path):
        os.makedirs(path)


def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(PATH_IMAGES):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(PATH_IMAGES, ""))
    return l_images


def create_detections():
    dic_model = load_dict_model(PATH_MODEL)
    images = find_image_names()
    m_border = load_model(dic_model["border"])

    with tqdm(total=len(images)) as j:
        for i in images:
            img = cv2.imread(os.path.join(PATH_IMAGES, i))
            mask, cnt = segmentation(img, m_border)
            find_circles(i, img, mask, cnt)
            j.update(1)


def load_dict_model(path):
    # gets all files inside the path
    files = os.listdir(path)
    model = {}

    model["model_h5"] = os.path.join(
        path, list(filter(lambda x: "classification" in x, files))[0]
    )

    model["border"] = os.path.join(
        path, list(filter(lambda x: "segmentation" in x, files))[0]
    )

    model["labels"] = ["Capped", "Eggs", "Honey", "Larves", "Nectar", "Other", "Pollen"]
    return model


def classify_images():
    images = sorted([os.path.join(PATH_IMAGES, i) for i in find_image_names()])

    find_image_detections = lambda i: ".".join(i.split(".")[:-1]) + ".npy"

    detections = [
        os.path.join(PATH_DETECTIONS, find_image_detections(i).replace(PATH_IMAGES, ""))
        for i in images
    ]

    dict_model = load_dict_model(PATH_MODEL)
    net = load_model(dict_model["model_h5"])

    with tqdm(total=len(images)) as t:
        for i, j in zip(images, detections):
            classify_image(i, j, dict_model["labels"], net, img_size, None)
            t.update(1)


def main():
    cross_plataform_directory()
    print("\nDetecting cells...")
    create_detections()
    print("\nClassifying cells...")
    classify_images()
    print("Done.")
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()