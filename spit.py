import cv2
import numpy as np
from pathlib import Path

def split_image(image_path, output_images):
    """
    Split an input image into 9 equal sections (3x3 grid) and save them as separate files.
    
    Args:
        image_path (Path): Path to the input image
        output_images (Path): Directory to save the split images
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    h, w = img.shape[:2]
    
    # Calculate dimensions for 3x3 grid
    h_step = h // 3
    w_step = w // 3
    
    # Define the sections for 3x3 grid
    sections = [
        # First row
        ((0, 0), (w_step, h_step)),               # Top-left
        ((w_step, 0), (2*w_step, h_step)),        # Top-middle
        ((2*w_step, 0), (w, h_step)),             # Top-right
        
        # Second row
        ((0, h_step), (w_step, 2*h_step)),        # Middle-left
        ((w_step, h_step), (2*w_step, 2*h_step)), # Center
        ((2*w_step, h_step), (w, 2*h_step)),      # Middle-right
        
        # Third row
        ((0, 2*h_step), (w_step, h)),            # Bottom-left
        ((w_step, 2*h_step), (2*w_step, h)),     # Bottom-middle
        ((2*w_step, 2*h_step), (w, h))           # Bottom-right
    ]
    
    # Process each section
    for idx, ((x1, y1), (x2, y2)) in enumerate(sections):
        try:
            section_img = img[y1:y2, x1:x2]
            section_img = cv2.resize(section_img, (640, 640))
            
            output_image_path = output_images / f"{image_path.stem}_{idx}.jpg"
            cv2.imwrite(str(output_image_path), section_img)
        except Exception as e:
            print(f"Error processing section {idx} of {image_path}: {e}")

def process_split(dataset_path, split):
    """
    Process all images in a dataset split.
    
    Args:
        dataset_path (str): Path to the dataset root directory
        split (str): Split name ('train' or 'valid')
    """
    input_path = Path(dataset_path) / split
    output_path = Path(dataset_path) / f"{split}_split"
    
    input_images = input_path / "images"
    output_images = output_path / "images"
    
    # Create output directory if it doesn't exist
    output_images.mkdir(parents=True, exist_ok=True)
    
    # Process each image in the input directory
    image_count = 0
    for image_path in input_images.glob("*.jpg"):
        try:
            split_image(image_path, output_images)
            image_count += 1
            if image_count % 10 == 0:  # Progress update every 10 images
                print(f"Processed {image_count} images...")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"Finished processing {image_count} images in {split} split")

if __name__ == "__main__":
    dataset_path = "Bee_siteA"
    process_split(dataset_path, "train")
    # Uncomment to process validation split
    # process_split(dataset_path, "valid")