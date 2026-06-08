import numpy as np
import cv2
import time

# --- 1. CẤU HÌNH DỮ LIỆU CỐ ĐỊNH (Tọa độ Robot World) ---

# Tọa độ Thế giới/Robot (X, Y) - cm. Dữ liệu này KHÔNG ĐỔI.
# Nếu bạn thay đổi vị trí vật lý của robot, bạn phải cập nhật tọa độ này.
WORLD_POINTS = np.array([
    [-10.5, -24.5], # P1
    [-10.5, -19.5],  # P2
    [34.5, -24.5], # P3
    [34.5, -19.5]  # P4
], dtype=np.float32)

# --- 2. CẤU HÌNH CAMERA & QUY TRÌNH ---

CAMERA_ID = 0    # ID camera (Cần điều chỉnh nếu sai)
NUM_POINTS = 4         # Số điểm cần thiết cho Homography
WINDOW_NAME = "Homography Quick Calibrator"
OUTPUT_FILE = 'homography_matrix.npy'

# Biến toàn cục để lưu trữ tọa độ pixel
pixel_points = []

def mouse_callback(event, u, v, flags, param):
    """
    Hàm được gọi khi có sự kiện chuột: Ghi lại tọa độ pixel.
    """
    global pixel_points
    
    if event == cv2.EVENT_LBUTTONDOWN and len(pixel_points) < NUM_POINTS:
        # Ghi lại tọa độ pixel
        pixel_points.append((u, v))
        print(f"✅ Đã ghi lại điểm {len(pixel_points)}: (u={u}, v={v})")
        
        # Cần vẽ lại hình ảnh để đánh dấu điểm mới
        img_display = param.copy() 
        draw_points(img_display)
        cv2.imshow(WINDOW_NAME, img_display)


def draw_points(frame):
    """Vẽ các điểm đã ghi lại lên frame"""
    for i, (pu, pv) in enumerate(pixel_points):
        cv2.circle(frame, (pu, pv), 5, (0, 255, 255), -1) 
        cv2.putText(frame, f"P{i+1}", (pu + 10, pv + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def run_calibration_tool():
    """Chạy công cụ hiệu chuẩn chính."""
    global pixel_points, WORLD_POINTS

    cap = cv2.VideoCapture(CAMERA_ID)
    

    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở camera ID {CAMERA_ID}. Vui lòng kiểm tra lại ID hoặc kết nối.")
        return

    cv2.namedWindow(WINDOW_NAME)
    
    # Pass frame cho mouse_callback
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, None)
    
    print("\n--- BẮT ĐẦU HIỆU CHUẨN HOMOGRAPHY ---")
    print(f"1. Hãy click theo đúng thứ tự (P1 -> P4) vào 4 điểm tham chiếu cố định.")
    print(f"2. Cần click {NUM_POINTS} điểm.")
    print("3. Nhấn ESC hoặc 'q' để thoát.")
    
    # Vòng lặp chính để hiển thị video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận frame. Exiting ...")
            break
        
        # Sao chép frame và cập nhật tham số cho mouse_callback
        img_display = frame.copy()
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, img_display)
        
        draw_points(img_display)
        cv2.imshow(WINDOW_NAME, img_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): 
            break

        # Nếu đã lấy đủ điểm, thoát vòng lặp
        if len(pixel_points) >= NUM_POINTS:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- 3. TÍNH TOÁN VÀ LƯU MA TRẬN H ---
    
    if len(pixel_points) == NUM_POINTS:
        print("\n--- BƯỚC 2: TÍNH TOÁN MA TRẬN H MỚI ---")
        
        pixel_points_np = np.array(pixel_points, dtype=np.float32)
        
        try:
            # Tính toán Homography
            H_NEW, mask = cv2.findHomography(pixel_points_np, WORLD_POINTS, cv2.RANSAC, 5.0)
            
            print("Ma trận Homography H mới đã tính:\n", H_NEW)
            
            # Lưu Ma trận H vào file
            np.save(OUTPUT_FILE, H_NEW) 
            print(f"\n✅ THÀNH CÔNG! Ma trận H mới đã được lưu vào file '{OUTPUT_FILE}'")
            print("   Bạn có thể khởi động lại Node Vision Processor của mình.")
            
        except Exception as e:
            print(f"❌ LỖI TÍNH TOÁN: Không thể tính toán Homography. {e}")
            print("   Kiểm tra xem 4 điểm có thẳng hàng hoặc quá gần nhau không.")
    
    else:
        print("Không đủ điểm để tính toán Homography. Vui lòng thử lại.")


if __name__ == "__main__":
    run_calibration_tool()
