# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import multiprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import shutil
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Image dimensions
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Paths
PATH_IMGS = "../original_images/"
ANNOTATIONS_FILE = "./predata_txt"
OUT_DATASET = "../dataset_train"
PATH_MODEL = "./model/classification.h5"
PATH_TRAIN = os.path.join(OUT_DATASET, "train")
PATH_VAL = os.path.join(OUT_DATASET, "validation")

# Configuration
BATCH_SIZE = 32
MAX_SAMPLES_CLASS = 50000
LABELS = {
    "Capped": 0,
    "Eggs": 1,
    "Honey": 2,
    "Larva": 3,
    "Nectar": 4,
    "Other": 5,
    "Pollen": 6
}

def create_folder(path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def create_dataset_structure():
    """Create all necessary folders for the dataset"""
    # Remove existing dataset folder if it exists
    shutil.rmtree(os.path.abspath(OUT_DATASET), ignore_errors=True)
    
    # Create main dataset folders
    create_folder(PATH_TRAIN)
    create_folder(PATH_VAL)
    
    # Create folders for each class in both train and validation
    for class_name in LABELS.keys():
        create_folder(os.path.join(PATH_TRAIN, class_name))
        create_folder(os.path.join(PATH_VAL, class_name))
    
    print("Created folders for all classes:")
    for class_name in LABELS.keys():
        print(f"- {class_name}")

def extract_cell(image, x, y, r):
    try:
        x, y = int(x), int(y)
        r = int(r)
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(image.shape[1], x + r)
        y2 = min(image.shape[0], y + r)
        cropped = image[y1:y2, x1:x2]
        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
            return resized
        return None
    except Exception as e:
        print(f"Error in extract_cell: {str(e)}")
        return None

def load_annotations(annotations_folder):
    """Load annotations from .txt files"""
    annotations = []
    
    # List all annotation files in the folder
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith('.txt'):
            image_name = os.path.splitext(annotation_file)[0]
            annotation_path = os.path.join(annotations_folder, annotation_file)
            
            # Read the corresponding annotation file
            if os.path.exists(annotation_path):
                with open(annotation_path, "r") as file:
                    for line in file:
                        parts = line.strip().split(",")
                        x, y, r = map(float, parts[:3])
                        class_id = int(parts[3])
                        annotations.append((image_name, x, y, r, class_id))
    
    return annotations

def save_image(data):
    """Save image to the specified path"""
    if data[0] is not None:
        cv2.imwrite(data[1], data[0])

def create_dataset():
    """Create the dataset using verified annotations"""
    # Create all necessary folders first
    create_dataset_structure()
    
    annotations = load_annotations(ANNOTATIONS_FILE)

    # Group annotations by image
    images_dict = {}
    for ann in annotations:
        if ann[0] not in images_dict:
            images_dict[ann[0]] = []
        images_dict[ann[0]].append(ann[1:])

    # Initialize dict_classes using the LABELS values
    dict_classes = {class_id: [] for class_id in LABELS.values()}

    print("\nProcessing annotations...")
    with tqdm(total=len(images_dict)) as t:
        for image_name, annots in images_dict.items():
            image_path = os.path.join(PATH_IMGS, image_name + ".jpg")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Process each annotation for this image
            for x, y, r, class_id in annots:
                cell = extract_cell(image, x, y, r)
                if cell is not None:
                    dict_classes[class_id].append((cell, (x, y, r)))

            t.update(1)

    print("\nSaving images to dataset...")
    # Create a reverse mapping from class_id to class_name
    id_to_name = {v: k for k, v in LABELS.items()}
    
    for class_id, samples in dict_classes.items():
        if not samples:
            continue
            
        cl_name = id_to_name[class_id]

        # Limit samples and split into train/validation
        np.random.shuffle(samples)
        samples = samples[:MAX_SAMPLES_CLASS]
        split_idx = int(len(samples) * 0.8)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # Save images for training and validation
        for dataset, dataset_samples, folder in [("train", train_samples, PATH_TRAIN), 
                                               ("validation", val_samples, PATH_VAL)]:
            print(f"Saving {len(dataset_samples)} images for {cl_name} in {dataset}")
            for idx, (cell, coords) in enumerate(dataset_samples):
                save_path = os.path.join(
                    folder, cl_name, 
                    f"{cl_name}_{idx}_{int(coords[0])}_{int(coords[1])}.jpg"
                )
                save_image((cell, save_path))

def create_model():
    """Create a CNN model for cell classification with smaller input size"""
    model = Sequential([
        # First Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Second Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),  # Reduced from 512
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(LABELS), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    """Train the model using the generated dataset"""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        PATH_TRAIN, 
        target_size=(64, 64),
        batch_size=BATCH_SIZE
    )

    val_generator = val_datagen.flow_from_directory(
        PATH_VAL,
        target_size=(64, 64),
        batch_size=BATCH_SIZE
    )

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1
        ),
        ModelCheckpoint(
            PATH_MODEL, 
            verbose=1, 
            save_best_only=True,
            monitor='val_accuracy'
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        )
        ]

    model = create_model()
    model.summary()

    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = val_generator.samples // BATCH_SIZE

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    print("\nTraining completed.")

def verify_dataset():
    """
    Verify dataset integrity by checking if all directories and files exist
    Returns True if dataset is valid, False otherwise
    """
    # Check main directories
    if not os.path.exists(PATH_TRAIN) or not os.path.exists(PATH_VAL):
        return False
    
    # Check class directories in both train and validation
    for class_name in LABELS.keys():
        train_class_path = os.path.join(PATH_TRAIN, class_name)
        val_class_path = os.path.join(PATH_VAL, class_name)
        if not os.path.exists(train_class_path) or not os.path.exists(val_class_path):
            return False
    
    return True

def main():
    #print("\nCreating Dataset...\n")
   # create_dataset()
    
    print("\nDataset created. You can now check the images in these folders:")
    print(f"Training data: {PATH_TRAIN}")
    print(f"Validation data: {PATH_VAL}")
    print("\nYou can remove any unwanted images now.")
    input("\nPress Enter when you're ready to continue with training...")
    
    # Verify dataset integrity before proceeding
    if not verify_dataset():
        print("\nError: Dataset structure appears to be invalid or corrupted.")
        print("Creating fresh dataset structure...")
        create_dataset_structure()
        print("\nPlease run the program again to recreate the dataset.")
        input("\nPress Enter to exit...")
        return

    print("\n\nStarting Training...\n\n")
    train()
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()