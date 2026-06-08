#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
import numpy as np

# --- CẤU HÌNH CỐ ĐỊNH ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
Y_FIXED = -22.13  # <--- Tọa độ Y cố định bạn mong muốn
X_HOME, Y_HOME, Z_HOME, THETA_HOME = -0.0, -12.0, 14.0, -90.0

# Các thông số khác giữ nguyên như cũ
SERVO_MAP = {'theta1': 9, 'theta2': 10, 'theta3': 11, 'theta4': 2, 'gripper': 24}
SERVO_ZERO_PULSES = {'theta1': 1500, 'theta2': 1100, 'theta3': 1550, 'theta4': 2100, 'gripper': 1250}
PULSE_PER_DEGREE = 2000 / 180.0
PWM_MỞ = 900
PWM_GẮP = {'blue': 1300, 'pink': 1200, 'yellow': 1200, 'red': 1200}
BOX_POSITIONS = {
    'blue':   {'x': 12.58, 'y': -15.0,  'z': 15.0, 'theta': -60.0},
    'pink':   {'x': 13.59, 'y': -4.95,  'z': 15.0, 'theta': -90.0},
    'yellow': {'x': 27.63, 'y': -15.95, 'z': 15.0, 'theta': -40.0},
    'red':    {'x': 26.99, 'y': -7.23,  'z': 15.0, 'theta': -40.0}
}

def inverse_kinematics_5dof(xe, ye, ze, theta_target_rad, a2, a3, a4, d1, elbow_config="down"):
    theta1 = np.arctan2(ye, xe)
    c1, s1 = np.cos(theta1), np.sin(theta1)
    nx = xe * c1 + ye * s1 - a4 * np.cos(theta_target_rad)
    ny = ze - d1 - a4 * np.sin(theta_target_rad)
    cos_theta3 = np.clip((nx**2 + ny**2 - a2**2 - a3**2) / (2 * a2 * a3), -1.0, 1.0)
    sin_theta3 = -np.sqrt(1 - cos_theta3**2)
    theta3 = np.arctan2(sin_theta3, cos_theta3)
    D = (a3 * cos_theta3 + a2)**2 + a3**2 * sin_theta3**2
    theta2 = np.arctan2(ny * (a3 * cos_theta3 + a2) - a3 * sin_theta3 * nx, nx * (a3 * cos_theta3 + a2) + a3 * sin_theta3 * ny)
    theta4 = theta_target_rad - (theta2 + theta3)
    return theta1, theta2, theta3, theta4

class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control_node')
        self.subscription = self.create_subscription(String, 'arm_command', self.listener_callback, 10)
        self.status_pub = self.create_publisher(String, 'arm_status', 10)
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        self.l1, self.l2, self.l3, self.l4 = 12.5, 15.5, 12.0, 14.0
        self.go_home()

    def send_status(self, status):
        self.status_pub.publish(String(data=status))

    def move_to_ik(self, x, y, z, theta_deg, gripper_p, speed=1000):
        try:
            t1, t2, t3, t4 = inverse_kinematics_5dof(x, y, z, np.deg2rad(theta_deg), self.l2, self.l3, self.l4, self.l1)
            angles = {'theta1': np.degrees(t1)+90, 'theta2': np.degrees(t2)-90, 'theta3': np.degrees(t3)+90, 'theta4': np.degrees(t4)+90}
            cmd = ""
            for name, ang in angles.items():
                zero = SERVO_ZERO_PULSES[name]
                factor = -1 if name in ['theta4', 'theta2'] else 1
                pulse = int(zero + (ang * PULSE_PER_DEGREE * factor))
                cmd += f"#{SERVO_MAP[name]}P{max(500, min(2500, pulse))}S{speed}"
            cmd += f"#{SERVO_MAP['gripper']}P{gripper_p}S{speed}\r\n"
            self.ser.write(cmd.encode())
            time.sleep(speed/1000.0 + 0.2)
        except: pass

    def go_home(self):
        self.move_to_ik(X_HOME, Y_HOME, Z_HOME, THETA_HOME, PWM_MỞ, 1500)
        self.send_status("FREE")

    def listener_callback(self, msg):
        parts = msg.data.split(',')
        if parts[0] != "PICK": return
        
        color = parts[1].lower()
        # Bỏ qua y từ camera, dùng Y_FIXED
        x_target = float(parts[2])
        
        self.get_logger().info(f"🦾 Đang gắp {color} tại X={x_target}, Y cố định={Y_FIXED}")
        self.send_status("BUSY")

        # Thực thi kịch bản với Y cố định
        try:
            # 1. Chờ cao
            self.move_to_ik(x_target, Y_FIXED, 21.38, -60.0, PWM_MỞ)
            # 2. Hạ gắp (Z=10.0)
            self.move_to_ik(x_target, Y_FIXED, 10.0, -60.0, PWM_MỞ, speed=500)
            # 3. Đóng kẹp
            p_close = PWM_GẮP.get(color, 1200)
            self.move_to_ik(x_target, Y_FIXED, 10.0, -60.0, p_close, speed=300)
            # 4. Nâng lên
            self.move_to_ik(x_target, Y_FIXED, 21.38, -60.0, p_close)
            # 5. Phân loại (Sử dụng tọa độ Box của bạn)
            box = BOX_POSITIONS.get(color, BOX_POSITIONS['blue'])
            self.move_to_ik(17.56, -10.14, 21.38, -60.0, p_close) # Transit
            self.move_to_ik(box['x'], box['y'], box['z'], box['theta'], p_close)
            self.move_to_ik(box['x'], box['y'], 10.0, box['theta'], PWM_MỞ, speed=500) # Thả
        finally:
            self.go_home()

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ArmControlNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
