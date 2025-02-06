import pupil_apriltags as apriltag
import cv2
import os
import numpy as np

def find_tag_centers(tags):
    """หาจุดกึ่งกลางของแท็กทั้งหมด"""
    tag_centers = {}
    for tag in tags:
        tag_centers[tag.tag_id] = np.mean(tag.corners, axis=0)
    return tag_centers

def draw_lines_on_frame(undistorted, tags):
    """วาดเส้นบนเฟรม"""
    frame_with_lines = undistorted.copy()
    tag_centers = find_tag_centers(tags)
    for tag in tags:
        # วาดเส้นรอบ AprilTag
        for j in range(4):
            cv2.line(frame_with_lines, tuple(tag.corners[j].astype(int)), 
                     tuple(tag.corners[(j+1)%4].astype(int)), (0, 255, 0), 2)
        cv2.putText(frame_with_lines, str(tag.tag_id), 
                    tuple(tag.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if len(tag_centers) >= 2:
        pt1 = tuple(tag_centers[list(tag_centers.keys())[0]].astype(int))
        pt2 = tuple(tag_centers[list(tag_centers.keys())[1]].astype(int))
        # เส้นสีส้มสำหรับกรอบบน
        offset_y_upper = 105
        offset_x = 90
        new_pt1 = (pt1[0] + offset_x, pt1[1] - offset_y_upper)
        new_pt2 = (pt2[0] - 125, pt2[1] - offset_y_upper)
        cv2.line(frame_with_lines, new_pt1, new_pt2, (0, 165, 255), 3)
        # เส้นสีส้มสำหรับกรอบล่าง
        offset_y_lower = 585
        new_pt3 = (pt1[0] + offset_x, pt1[1] - offset_y_lower)
        new_pt4 = (pt2[0] - 125, pt2[1] - offset_y_lower)
        cv2.line(frame_with_lines, new_pt3, new_pt4, (0, 165, 255), 3)
        # เส้นกรอบข้าง
        cv2.line(frame_with_lines, new_pt1, new_pt3, (0, 165, 255), 3)
        cv2.line(frame_with_lines, new_pt2, new_pt4, (0, 165, 255), 3)
    return frame_with_lines

def draw_lines_and_crop(undistorted, tags):
    """วาดเส้นและครอปภาพ"""
    frame_with_lines = undistorted.copy()
    if len(tags) >= 2:
        # หาจุดกึ่งกลางของแท็ก
        tag_centers = {}
        for tag in tags:
            tag_centers[tag.tag_id] = np.mean(tag.corners, axis=0)
        # เรียงลำดับตำแหน่งแท็ก
        sorted_tag_ids = sorted(tag_centers.keys())
        pt1 = tuple(tag_centers[sorted_tag_ids[0]].astype(int))
        pt2 = tuple(tag_centers[sorted_tag_ids[1]].astype(int))
        try:
            # คำนวณพิกัดมุมทั้ง 4 มุม
            x_left_top = min(pt1[0], pt2[0]) 
            x_right_top = max(pt1[0], pt2[0])
            y_top = min(pt1[1], pt2[1])
            
            y_bottom = max(pt1[1], pt2[1])
            y_top = y_top - 585
            y_bottom = y_bottom - 105
            x_left_top = x_left_top - 125
            x_right_top = x_right_top + 90
            
            # ตรวจสอบว่าพิกัดไม่ติดลบและไม่เกินขนาดภาพ
            height, width = undistorted.shape[:2]
            y_top = max(0, y_top)
            y_bottom = min(height, y_bottom)
            x_left_top = max(0, x_left_top)
            x_right_top = min(width, x_right_top)
            
            if y_bottom > y_top and x_right_top > x_left_top:
                cropped_image = undistorted[y_top:y_bottom, x_left_top:x_right_top]
                return frame_with_lines, cropped_image
            else:
                print("พื้นที่ครอปไม่ถูกต้อง")
                return frame_with_lines, None
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการครอปภาพ: {e}")
            return frame_with_lines, None
    return undistorted, None

def crop_with_fixed_coordinates(image):
    """ครอปภาพตามพิกัดที่กำหนดไว้"""
    try:
        height, width = image.shape[:2]
        
        # กำหนดพิกัดสำหรับการครอป (ปรับค่าตามต้องการ)
        x_left_top = 140    # พิกัด x เริ่มต้น
        x_right_top = 1150  # พิกัด x สิ้นสุด
        y_top = 115        # พิกัด y เริ่มต้น
        y_bottom = 595     # พิกัด y สิ้นสุด
        
        # ตรวจสอบว่าพิกัดไม่ติดลบและไม่เกินขนาดภาพ
        x_left_top = max(0, x_left_top)
        x_right_top = min(width, x_right_top)
        y_top = max(0, y_top)
        y_bottom = min(height, y_bottom)
        
        if y_bottom > y_top and x_right_top > x_left_top:
            cropped_image = image[y_top:y_bottom, x_left_top:x_right_top]
            return cropped_image
        else:
            print("พิกัดที่กำหนดไว้ไม่ถูกต้อง")
            return None
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการครอปภาพด้วยพิกัดที่กำหนด: {e}")
        return None

# ตั้งค่าตัวตรวจจับ AprilTag
detector = apriltag.Detector(
    families='tag36h11',
    nthreads=4,
    quad_decimate=0.5,
    quad_sigma=0.5,
    refine_edges=True,
    debug=False
)

# กำหนดโฟลเดอร์ input และ output
input_folder = 'pic'  # โฟลเดอร์ที่เก็บรูปต้นฉบับ
output_folder = 'cropped_pics'  # โฟลเดอร์สำหรับเก็บรูปที่ครอปแล้ว

# สร้างโฟลเดอร์ output ถ้ายังไม่มี
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ประมวลผลทุกรูปในโฟลเดอร์
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # เฉพาะไฟล์รูปภาพ
        # อ่านรูปภาพ
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"ไม่สามารถอ่านรูป {filename} ได้")
            continue
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ตรวจจับ AprilTag
        results = detector.detect(gray)

        output_filename = os.path.splitext(filename)[0] + '_cropped' + os.path.splitext(filename)[1]
        output_path = os.path.join(output_folder, output_filename)

        if len(results) > 0:
            print(f"{filename}: พบ {len(results)} AprilTag")

            # วาดเส้นและครอปภาพ
            frame_with_lines, cropped_image = draw_lines_and_crop(image, results)
            
            if cropped_image is not None:
                # ปรับขนาดรูปเป็น 1280x720
                resized_image = cv2.resize(cropped_image, (1280, 720), interpolation=cv2.INTER_AREA)
                
                # บันทึกภาพที่ครอปแล้ว
                cv2.imwrite(output_path, resized_image)
                print(f"บันทึกภาพที่ครอปและปรับขนาดแล้วที่: {output_path}")
            else:
                print(f"{filename}: ไม่สามารถครอปภาพได้")
        else:
            print(f"{filename}: ไม่พบ AprilTag - ทำการครอปด้วยพิกัดที่กำหนดไว้")
            cropped_image = crop_with_fixed_coordinates(image)
            if cropped_image is not None:
                # ปรับขนาดรูปเป็น 1280x720
                resized_image = cv2.resize(cropped_image, (1280, 720), interpolation=cv2.INTER_AREA)
                
                # บันทึกภาพที่ครอปแล้ว
                cv2.imwrite(output_path, resized_image)
                print(f"บันทึกภาพที่ครอปและปรับขนาดแล้วที่: {output_path}")
            else:
                print(f"{filename}: ไม่สามารถครอปภาพด้วยพิกัดที่กำหนดได้")

print("เสร็จสิ้นการประมวลผลทุกรูปภาพ")