import serial
import time
import cv2
import numpy as np
import pupil_apriltags as apriltag
import os

def setup_camera():
    """ตั้งค่ากล้องโดยลองทุก index จนกว่าจะเจอ"""
    index = 0
    while True:
        camera = cv2.VideoCapture(index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลดขนาด buffer
        camera.set(cv2.CAP_PROP_FPS, 60)  # เพิ่ม FPS
        
        # ตรวจสอบว่ากล้องเปิดได้หรือไม่
        if not camera.isOpened():
            print(f"ไม่สามารถเปิดกล้องที่ index {index} ได้ กำลังลองใหม่...")
            camera.release()

            time.sleep(1)
            continue
        
        # ทดสอบการเชื่อมต่อโดยการอ่านเฟรมแรก
        ret, frame = camera.read()
        if not ret:
            print(f"ไม่สามารถอ่านข้อมูลจากกล้องที่ index {index} ได้ กำลังลอง index ถัดไป...")
            camera.release()
            index += 1
            if index > 10:
                print("ไม่พบกล้องที่ใช้งานได้")
                return None
            time.sleep(1)
            continue
        
        print(f"กล้องพร้อมใช้งานที่ index {index}")
        return camera
    
def find_tag_centers(tags):
    """หาจุดกึ่งกลางของแท็ก"""
    tag_centers = {}
    for tag in tags:
        tag_centers[tag.tag_id] = np.mean(tag.corners, axis=0)
    return tag_centers

def detect_apriltags(frame, detector):
    """ตรวจจับ AprilTag ในเฟรม"""
    # ลดขนาดภาพก่อนประมวลผล
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    results = detector.detect(gray)
    
    # ปรับพิกัดกลับไปยังขนาดเดิม
    for tag in results:
        tag.corners *= (1/scale)
        tag.center *= (1/scale)
    return results

def create_directories():
    """สร้างโฟลเดอร์สำหรับเก็บภาพ"""
    os.makedirs('images/original/side_A', exist_ok=True)
    os.makedirs('images/cropped/side_A', exist_ok=True)
    os.makedirs('images/original/side_B', exist_ok=True)
    os.makedirs('images/cropped/side_B', exist_ok=True)
    
def draw_lines_on_frame(undistorted, tags):
    """วาดเส้นบนเฟรม"""
    # สร้างภาพสำเนาเพื่อวาดเส้น
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
        offset_y_upper = 110
        offset_x = 19
        new_pt1 = (pt1[0] + offset_x, pt1[1] - offset_y_upper)
        new_pt2 = (pt2[0] - 50, pt2[1] - offset_y_upper)
        cv2.line(frame_with_lines, new_pt1, new_pt2, (0, 165, 255), 3)

        # เส้นสีส้มสำหรับกรอบล่าง
        offset_y_lower = 620
        new_pt3 = (pt1[0] + offset_x, pt1[1] - offset_y_lower)
        new_pt4 = (pt2[0] - 50, pt2[1] - offset_y_lower)
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
            y_top = y_top - 620
            y_bottom = y_bottom - 110
            x_left_top = x_left_top - 50
            x_right_top = x_right_top + 25
            
            # ตรวจสอบว่าพิกัดไม่ติดลบและไม่เกินขนาดภาพ
            height, width = undistorted.shape[:2]
            y_top = max(0, y_top)
            y_bottom = min(height, y_bottom)
            x_left_top = max(0, x_left_top)
            x_right_top = min(width, x_right_top)
            
            # ตรวจสอบว่าพื้นที่ครอปมีขนาดมากกว่า 0
            if y_bottom > y_top and x_right_top > x_left_top:
                # ครอปภาพภายในสี่เหลี่ยม
                cropped_image = undistorted[y_top:y_bottom, x_left_top:x_right_top]
                return frame_with_lines, cropped_image
            else:
                print("พื้นที่ครอปไม่ถูกต้อง")
                return frame_with_lines, None
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการครอปภาพ: {e}")
            return frame_with_lines, None

    return undistorted, None

def check_serial_connection(max_attempts=1000, timeout=5):
    """
    Attempt to establish a serial connection with multiple retry attempts.
    
    Args:
        max_attempts (int): Maximum number of connection attempts
        timeout (int): Timeout for each connection attempt in seconds
    
    Returns:
        serial.Serial or None: Connected serial object or None if failed
    """
    for attempt in range(max_attempts):
        try:
            print(f"กำลังเชื่อมต่อพอร์ต Serial (พยายามครั้งที่ {attempt + 1}/{max_attempts})...")
            ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            print("เชื่อมต่อ Serial สำเร็จ")
            return ser
        except serial.SerialException as e:
            print(f"ไม่สามารถเชื่อมต่อ Serial: {e}")
            time.sleep(timeout)
    
    print("ไม่สามารถเชื่อมต่อ Serial หลังจากพยายาม {} ครั้ง".format(max_attempts))
    return None

def check_camera_connection(index=0, max_attempts=5):
    for attempt in range(max_attempts):
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            print(f"เชื่อมต่อกล้องที่ index {index} สำเร็จ")
            return camera
        time.sleep(1)
    print("ไม่สามารถเชื่อมต่อกล้องได้")
    return None

def capture_image(camera, camera_matrix, dist_coeffs, detector, current_side='side_A'):
    """ถ่ายและบันทึกภาพ"""
    camera.release()
    camera = setup_camera()
    ret, frame = camera.read()
    if ret:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Undistort frame
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        # Detect tags
        tags = detect_apriltags(undistorted, detector)
        
        if len(tags) >= 2:
            # วาดเส้นและครอปภาพ
            frame_with_lines, cropped_image = draw_lines_and_crop(undistorted, tags)
            
            # บันทึกภาพดั้งเดิมไม่มีการแก้ไข
            original_path = f"images/original/{current_side}/original_image_{timestamp}.jpg"
            cv2.imwrite(original_path, undistorted)
            print(f"บันทึกภาพต้นฉบับที่: {original_path}")
            
            # บันทึกภาพครอปพร้อมเส้น
            cropped_path = f"images/cropped/{current_side}/cropped_image_{timestamp}.jpg"
            cv2.imwrite(cropped_path, cropped_image)
            print(f"บันทึกภาพครอปที่: {cropped_path}")
        else:
            original_path = f"images/original/{current_side}/original_image_{timestamp}.jpg"
            cv2.imwrite(original_path, undistorted)
            print(f"บันทึกภาพต้นฉบับที่: {original_path}")
            print("ไม่พบแท็กครบ 2 แท็ก")
    else:
        print("ถ่ายภาพไม่สำเร็จ")
    return camera

def get_screen_resolution():
    """Get the primary screen resolution"""
    try:
        # Create a temporary window to get screen info
        cv2.namedWindow('temp', cv2.WINDOW_NORMAL)
        screen_width = cv2.getWindowImageRect('temp')[2]
        screen_height = cv2.getWindowImageRect('temp')[3]
        cv2.destroyWindow('temp')
        return screen_width, screen_height
    except:
        # Fallback resolution if unable to detect
        return 1920, 1080

def setup_display_window():
    """Set up the display window in fullscreen mode"""
    window_name = 'AprilTag Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return window_name

def main():
    current_side = 'side_A'
    capture_count = 0
    while True:
        try:
            # สร้างโฟลเดอร์
            create_directories()
            
            # โหลดข้อมูลการคาลิเบรทกล้อง
            with np.load('calibration_data/CalibrationMatrix_college_cpt.npz') as data:
                camera_matrix = data['Camera_matrix']
                dist_coeffs = data['distCoeff']

            # ตรวจสอบการเชื่อมต่อ Serial
            ser = None
            while ser is None:
                ser = check_serial_connection()
                if ser is None:
                    print("รอการเสียบพอร์ต Serial...")
                    time.sleep(5)  # รอ 5 วินาที ก่อนลองใหม่

            # ตรวจสอบการเชื่อมต่อกล้อง
            camera = None
            while camera is None:
                camera = setup_camera()
                if camera is None:
                    print("รอการเสียบกล้อง...")
                    time.sleep(5)  # รอ 5 วินาที ก่อนลองใหม่

            # ตั้งค่าตัวตรวจจับ AprilTag
            detector = apriltag.Detector(
                families='tag36h11',
                nthreads=4,
                quad_decimate=0.5,
                quad_sigma=0.5,
                refine_edges=True,
                debug=False
            )
            
            window_name = setup_display_window()

            # แสดงภาพพรีวิว AprilTag
            preview_started = False
            while True:
                try:
                    # ตรวจสอบการเชื่อมต่อกล้องในแต่ละรอบ
                    ret, frame = camera.read()
                    if not ret:
                        print("การอ่านเฟรมล้มเหลว กำลังรีเซ็ตกล้อง...")
                        camera.release()
                        camera = setup_camera()
                        if camera is None:
                            break  # ออกจากลูปหากไม่สามารถเชื่อมต่อกล้องได้
                        time.sleep(0.1)
                        continue

                    # ตรวจสอบการเชื่อมต่อ Serial
                    if ser.is_open:
                        try:
                            if ser.in_waiting > 0:
                                command = ser.readline().decode('utf-8').strip()
                                print(f"ได้รับคำสั่ง: {command}")
                                
                                if command == "START_PREVIEW":
                                    preview_started = True
                                    cv2.destroyAllWindows()
                                    break
                        except Exception as serial_error:
                            print(f"เกิดข้อผิดพลาดที่พอร์ต Serial: {serial_error}")
                            ser.close()
                            # รอการเสียบพอร์ตใหม่
                            while ser is None:
                                ser = check_serial_connection()
                                if ser is None:
                                    print("รอการเสียบพอร์ต Serial...")
                                    time.sleep(5)
                    else:
                        # พยายามเชื่อมต่อ Serial ใหม่
                        ser = check_serial_connection()
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    # ประมวลผลและแสดงภาพ
                    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
                    tags = detect_apriltags(undistorted, detector)
                    
                    display_frame = draw_lines_on_frame(undistorted.copy(), tags)
                    cv2.imshow(window_name, display_frame)
                    
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        cv2.destroyAllWindows()
                        break

                except Exception as e:
                    print(f"เกิดข้อผิดพลาด: {e}")
                    # พยายามรีเซ็ตการเชื่อมต่อ
                    if camera:
                        camera.release()
                    camera = setup_camera()
                    time.sleep(1)

            # โหมดรอรับคำสั่ง
            if preview_started:
                while True:
                    try:
                        if ser.is_open and ser.in_waiting > 0:
                            command = ser.readline().decode('utf-8').strip()
                            print(f"ได้รับคำสั่ง: {command}")
                            
                            if command == "CAMERA_CAPTURE":
                                camera = capture_image(camera, camera_matrix, dist_coeffs, detector, current_side)
                                ser.write(b'CAPTURE_DONE\n')
                                print(f"ถ่ายภาพเสร็จแล้ว ด้าน {current_side} ส่งการยืนยันกลับไป")
                                
                                # สลับด้านและนับจำนวนการถ่าย
                                capture_count += 1
                                if capture_count >= 2:
                                    current_side = 'side_A'  # รีเซ็ทกลับเป็นด้าน A
                                    capture_count = 0
                                else:
                                    current_side = 'side_B'
                        
                        # ตรวจสอบการเชื่อมต่อต่อเนื่อง
                        if not ser.is_open or not camera.isOpened():
                            print("การเชื่อมต่อขาดหาย กำลังรีเซ็ต...")
                            break
                    
                    except Exception as e:
                        print(f"เกิดข้อผิดพลาดในโหมดรอรับคำสั่ง: {e}")
                        break

        except KeyboardInterrupt:
            print("โปรแกรมถูกหยุด")
            break
        
        except Exception as e:
            print(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
            # รอสักครู่ก่อนลองใหม่
            time.sleep(5)
        
        finally:
            # ทำความสะอาดทรัพยากร
            if 'camera' in locals() and camera is not None:
                camera.release()
            if 'ser' in locals() and ser is not None:
                ser.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()