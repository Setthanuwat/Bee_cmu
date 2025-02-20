from ultralytics import YOLO
import os
from pathlib import Path
import cv2
from collections import defaultdict
from PIL import Image
import torch
import json

# Add configuration class to manage parameters
class DetectionConfig:
    def __init__(self):
        self.model_path = 'model_- 13 february 2025 17_40.pt'
        self.data_folder = "datafull"
        self.max_area_percentage = 0.2
        self.initial_conf = 0.35
        self.high_conf = 0.50
        self.iou_threshold = 0.15
        self.results_folder = "results_model_- 13 february 2025 17_40_2"
        
    def save_config(self, path):
        config_dict = {
            "model_path": self.model_path,
            "data_folder": self.data_folder,
            "max_area_percentage": self.max_area_percentage,
            "initial_conf": self.initial_conf,
            "high_conf": self.high_conf,
            "iou_threshold": self.iou_threshold,
            "results_folder": self.results_folder
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self, key, value)

def adjust_iou_threshold(bee_count):
    if bee_count >= 800:
        return 0.6
    elif 600 < bee_count < 800:
        return 0.4
    else:
        return 0.15

def adjust_initial_conf(bee_count):
    if bee_count >= 800:
        return 0.28
    elif 600 < bee_count < 800:
        return 0.30
    elif bee_count < 60:
        return 0.65
    else:
        return 0.35
    
def filter_boxes_by_area_and_nms(results, max_area_percentage=0.2, iou_threshold=0.1):
    filtered_results = []
    for result in results:
        img_height, img_width = result.orig_shape
        img_area = img_width * img_height
        boxes = result.boxes
        filtered_boxes = []
        
        boxes_for_nms = []
        confidences = []
        filtered_indices = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            area_percentage = box_area / img_area
            
            if area_percentage <= max_area_percentage:
                boxes_for_nms.append([x1, y1, x2, y2])
                confidences.append(box.conf[0])
                filtered_indices.append(i)
        
        if boxes_for_nms:
            boxes_tensor = torch.tensor(boxes_for_nms)
            conf_tensor = torch.tensor(confidences)
            # Use dynamic IOU threshold based on bee count
            nms_indices = torch.ops.torchvision.nms(boxes_tensor, conf_tensor, iou_threshold)
            
            for idx in nms_indices:
                filtered_boxes.append(boxes[filtered_indices[idx]])
        
        result.boxes = filtered_boxes
        filtered_results.append(result)
    
    return filtered_results

def detect_with_confidence(model, image_path, conf_threshold=0.5, max_area_percentage=0.2, iou_threshold=0.1):
    results = model(image_path, conf=conf_threshold)
    filtered_results = filter_boxes_by_area_and_nms(results, max_area_percentage, iou_threshold)
    
    bee_count = 0
    bg_count = 0
    for box in filtered_results[0].boxes:
        cls = int(box.cls[0])
        if cls == 1:
            bee_count += 1
        elif cls == 0:
            bg_count += 1
    
    return filtered_results, (bee_count, bg_count)

def merge_images_and_count_boxes(image_folder, output_path):
    all_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and "detected_" in f and "_cropped_" in f]
    
    file_groups = {}
    total_boxes_per_group = {}
    
    for f in all_files:
        base_name = f.split("_cropped_")[0].replace("detected_", "")
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(f)
    
    for base_name, image_files in file_groups.items():
        image_files.sort(key=lambda x: int(x.split("_cropped_")[-1].split(".")[0]))
        images = [Image.open(os.path.join(image_folder, f)) for f in image_files]
        width, height = images[0].size
        
        merged_image = Image.new('RGB', (width * 3, height * 3))
        
        for i, img in enumerate(images):
            x_offset = (i % 3) * width
            y_offset = (i // 3) * height
            merged_image.paste(img, (x_offset, y_offset))
        
        merged_image = merged_image.resize((1280, 720))
        output_file = os.path.join(output_path, f"{base_name}_merged.jpg")
        merged_image.save(output_file)
        print(f"Image merged and saved to {output_file}")
        
        # Read both Bee and BG counts
        with open(os.path.join(image_folder, "bee_counts.txt"), "r") as f:
            for line in f:
                if base_name in line:
                    bee_count = int(line.split(": ")[1])
                    if base_name not in total_boxes_per_group:
                        total_boxes_per_group[base_name] = {"Bee": 0, "BG": 0}
                    total_boxes_per_group[base_name]["Bee"] = bee_count
                    break
                    
        with open(os.path.join(image_folder, "bg_counts.txt"), "r") as f:
            for line in f:
                if base_name in line:
                    bg_count = int(line.split(": ")[1])
                    if base_name not in total_boxes_per_group:
                        total_boxes_per_group[base_name] = {"Bee": 0, "BG": 0}
                    total_boxes_per_group[base_name]["BG"] = bg_count
                    break
    
    return total_boxes_per_group

def merge_images(image_folder, output_path):
    all_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and "_cropped_" in f]
    
    file_groups = {}
    for f in all_files:
        base_name = f.split("_cropped_")[0]
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(f)
    
    for base_name, image_files in file_groups.items():
        image_files.sort(key=lambda x: int(x.split("_cropped_")[-1].split(".")[0]))
        images = [Image.open(os.path.join(image_folder, f)) for f in image_files]
        width, height = images[0].size
        
        merged_image = Image.new('RGB', (width * 3, height * 3))
        
        for i, img in enumerate(images):
            x_offset = (i % 3) * width
            y_offset = (i // 3) * height
            merged_image.paste(img, (x_offset, y_offset))
        
        merged_image = merged_image.resize((1280, 720))
        output_file = os.path.join(output_path, f"{base_name}_merged.jpg")
        merged_image.save(output_file)
        print(f"Image merged and saved to {output_file}")
        
def adjust_thresholds(bee_count):
    iou = adjust_iou_threshold(bee_count)
    conf = adjust_initial_conf(bee_count)
    return iou, conf
def main():
    # Initialize configuration
    config = DetectionConfig()
    
    # Create results directory
    Path(config.results_folder).mkdir(exist_ok=True)
    output_folder = os.path.join(config.results_folder, "output")
    Path(output_folder).mkdir(exist_ok=True)
    
    # Save initial configuration
    config.save_config(os.path.join(config.results_folder, "detection_config.json"))
    
    model = YOLO(config.model_path)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(config.data_folder) if f.lower().endswith(image_extensions)]

    print("\nStep 1: Initial detection...")
    bee_counts = defaultdict(int)
    bg_counts = defaultdict(int)
    
    # First pass - initial detection
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(config.data_folder, image_file)
        print(f"Processing image {i}/{len(image_files)}: {image_file}")
        
        base_name = image_file.split("_cropped_")[0]
        filtered_results, (bee_count, bg_count) = detect_with_confidence(
            model, image_path, config.initial_conf, config.max_area_percentage, config.iou_threshold
        )
        
        bee_counts[base_name] += bee_count
        bg_counts[base_name] += bg_count
        
        save_path = os.path.join(config.results_folder, f"detected_{image_file}")
        filtered_results[0].save(save_path)

    # Save initial counts
    with open(os.path.join(config.results_folder, "bee_counts.txt"), "w") as f:
        for name, count in bee_counts.items():
            f.write(f"{name}: {count}\n")
            
    with open(os.path.join(config.results_folder, "bg_counts.txt"), "w") as f:
        for name, count in bg_counts.items():
            f.write(f"{name}: {count}\n")

    print("\nStep 2: Merging images and counting boxes...")
    total_boxes_per_group = merge_images_and_count_boxes(config.results_folder, output_folder)

    print("\nStep 3: Checking for groups that need redetection...")
    need_redetection = False
    
    # Process groups based on total box count
    for base_name, counts in total_boxes_per_group.items():
        total_boxes = counts['Bee'] + counts['BG']
        current_bee_count = counts['Bee']
        
        # Determine which threshold category this falls into
        if total_boxes < 100:
            threshold_category = "low"
        elif 600 < total_boxes < 800:
            threshold_category = "medium"
        elif total_boxes >= 800:
            threshold_category = "high"
        else:
            continue  # Skip if within normal range
            
        need_redetection = True
        print(f"Re-detecting {base_name} ({threshold_category} count: {total_boxes} boxes)")
        
        related_images = [f for f in image_files if base_name in f]
        bee_sub_total = 0
        bg_sub_total = 0
        
        # Re-detect with adjusted parameters
        for image_file in related_images:
            image_path = os.path.join(config.data_folder, image_file)
            current_iou, current_conf = adjust_thresholds(current_bee_count)
            print(f"Using parameters: IOU={current_iou}, conf={current_conf}")
            
            filtered_results, (bee_count, bg_count) = detect_with_confidence(
                model, image_path, current_conf, config.max_area_percentage, current_iou
            )
            
            save_path = os.path.join(config.results_folder, f"detected_{image_file}")
            filtered_results[0].save(save_path)
            
            bee_sub_total += bee_count
            bg_sub_total += bg_count
        
        # Update counts for this group
        bee_counts[base_name] = bee_sub_total
        bg_counts[base_name] = bg_sub_total

    if need_redetection:
        print("\nSaving updated box counts...")
        with open(os.path.join(config.results_folder, "bee_counts.txt"), "w") as f:
            for name, count in bee_counts.items():
                f.write(f"{name}: {count}\n")
                
        with open(os.path.join(config.results_folder, "bg_counts.txt"), "w") as f:
            for name, count in bg_counts.items():
                f.write(f"{name}: {count}\n")
        
        print("\nStep 4: Merging images again with new detections...")
        total_boxes_per_group = merge_images_and_count_boxes(config.results_folder, output_folder)
    
    # Final merge of images
    merge_images(config.results_folder, output_folder)
    
    # Save final statistics
    final_stats = {
        "bee_counts": dict(bee_counts),
        "bg_counts": dict(bg_counts),
        "final_parameters": {
            "max_area_percentage": config.max_area_percentage,
            "initial_conf": config.initial_conf,
            "high_conf": config.high_conf,
            "base_iou_threshold": config.iou_threshold
        }
    }
    
    with open(os.path.join(config.results_folder, "final_stats.json"), "w") as f:
        json.dump(final_stats, f, indent=4)
    
    print("\nProcessing completed!")
    print("Final box counts per group:")
    for base_name, counts in total_boxes_per_group.items():
        print(f"{base_name}:")
        print(f"  Bee: {counts['Bee']} boxes")
        print(f"  BG: {counts['BG']} boxes")
        print(f"  Total: {counts['Bee'] + counts['BG']} boxes")
        
if __name__ == "__main__":
    main()