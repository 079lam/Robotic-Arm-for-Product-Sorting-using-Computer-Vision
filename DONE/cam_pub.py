#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage # Dùng ảnh nén
import cv2
from rclpy.qos import qos_profile_sensor_data # QoS tối ưu cho sensor

class ImagePublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        # Sử dụng qos_profile_sensor_data để giảm độ trễ (Best Effort)
        self.publisher_ = self.create_publisher(CompressedImage, 'image_raw/compressed', qos_profile_sensor_data)
        self.timer = self.create_timer(0.05, self.timer_callback) # Tăng lên ~20 FPS
        self.cap = cv2.VideoCapture(0)
        
        # Giảm độ phân giải để tăng tốc độ xử lý nếu cần
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            # Nén ảnh trực tiếp bằng OpenCV, không cần CvBridge cho CompressedImage
            msg.data = cv2.imencode('.jpg', frame)[1].tobytes()
            
            self.publisher_.publish(msg)
            # self.get_logger().info('Publishing compressed frame')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher("topic_webcam_pub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
