#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO

class ConveyorVisionProcessor(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 1. Cấu hình Model YOLO -
        model_path = os.path.expanduser('~/ros2_ws/src/cam_sub/best.pt')
        self.model = YOLO(model_path)
        
        # 2. Homography
        h_matrix_path = os.path.expanduser('~/ros2_ws/src/cam_sub/homography_matrix.npy')
        self.H = np.load(h_matrix_path) if os.path.exists(h_matrix_path) else None

        # 3. Trạng thái điều khiển
        self.arm_free = True        
        self.waiting_for_robot = False 
                
        self.frame_count = 0
        self.process_every_n_frames = 2 # Chỉ chạy AI mỗi 2 khung hình (tăng FPS)

        # 4. ROS setup
        self.image_sub = self.create_subscription(
            CompressedImage, 'image_raw/compressed', self.listener_callback, 
            qos_profile_sensor_data) # Giữ qos_profile_sensor_data để giảm trễ
        
        self.status_sub = self.create_subscription(String, 'arm_status', self.status_callback, 10)
        self.target_pub = self.create_publisher(String, 'arm_command', 10)

        self.get_logger().info("🚀 Node Vision đã tối ưu chống lag khởi chạy...")

    def status_callback(self, msg):
        if msg.data.upper() == "FREE":
            self.arm_free = True
            self.waiting_for_robot = False

    def pixel_to_world(self, u, v):
        if self.H is None: return (0.0, 0.0)
        point = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)
        world_point = np.dot(self.H, point)
        scale = world_point[2][0]
        return (round(world_point[0][0]/scale, 2), round(world_point[1][0]/scale, 2))

    def listener_callback(self, data):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return # Bỏ qua khung hình này để CPU nghỉ ngơi

        # Giải mã ảnh
        np_arr = np.frombuffer(data.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        # Tối ưu 2: Giảm kích thước ảnh đầu vào cho AI (giúp YOLO chạy nhanh gấp đôi)
        # 640x480 -> 320x240
        input_img = cv2.resize(image, (320, 240))

        # Chạy AI (Sử dụng stream=True để tối ưu bộ nhớ)
        results = self.model.predict(input_img, conf=0.6, verbose=False, imgsz=320)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, clss):
                # Lưu ý: Do đã resize ảnh xuống 1/2, ta phải nhân tọa độ tâm lên 2
                u_center = int((box[0] + box[2])) # (box[0]+box[2])/2 * 2
                v_center = int((box[1] + box[3]))
                
                x_real, y_real = self.pixel_to_world(u_center, v_center)
                label = self.model.names[int(cls)]

                # Vùng kích hoạt (mở rộng một chút để bù trễ lag)
                if (0 <= x_real <= 0.5) and (-24.5 <= y_real <= -19.5):
                    if self.arm_free and not self.waiting_for_robot:
                        self.target_pub.publish(String(data=f"PICK,{label},{x_real},{y_real},0"))
                        self.arm_free = False
                        self.waiting_for_robot = True
                        self.get_logger().info(f"🎯 Gửi lệnh: {label} X:{x_real}")

                # Vẽ hiển thị (vẽ trên ảnh gốc để nét)
                cv2.rectangle(image, (u_center-20, v_center-20), (u_center+20, v_center+20), (0,255,0), 2)

        # Tối ưu 3: Chỉ hiển thị ảnh khi cần thiết (cv2.imshow tiêu tốn nhiều CPU)
        cv2.imshow("Optimized Vision", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ConveyorVisionProcessor("vision_processor_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
