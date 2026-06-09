# A 5-DOF Robotic Arm System for Product Classification Using Computer Vision

This repository contains the source code, control nodes, and vision pipeline for an automated product sorting system. The project integrates a physical 5-Degree-of-Freedom (5-DOF) robotic arm, a conveyor belt mechanism, and an edge-computing vision subsystem to identify and classify objects based on physical attributes (color, shape, and size) in real time.

## 🚀 Key Features
* **Real-Time Object Detection:** Leverages the **YOLOv8** framework combined with **OpenCV** for low-latency feature extraction and product classification on a moving conveyor belt.
* **Spatial Coordinate Mapping:** Implements custom calibration scripts to compute a **Homography Matrix**, seamlessly transforming 2D camera pixel coordinates into accurate 3D real-world robot operational coordinates.
* **Distributed Robot Control:** Built entirely on the **ROS2 (Humble)** ecosystem, utilizing native Publisher/Subscriber nodes to handle data streaming and coordinate communication.
* **Kinematics & Motion Profiling:** Solves Forward and Inverse Kinematics for the 5-DOF configuration, implementing smooth trajectory planning to optimize stationary pick-and-place routines.

## 🛠️ Tech Stack & Hardware
* **Framework/OS:** ROS2 (Humble), Ubuntu 22.04 LTS
* **Target Hardware:** Raspberry Pi 5 (Edge Deployment), 5-DOF Robotic Arm, Conveyor Belt Setup
* **Languages:** Python
* **Primary Libraries:** OpenCV, YOLOv8 (Ultralytics), NumPy

## 📁 Repository Structure
```text
├── DONE/
│   ├── cam_pub.py                # ROS2 Node: Publishes the raw camera video stream
│   ├── cam_sub.py                # ROS2 Node: Subscribes to the stream and triggers YOLOv8 inference
│   ├── tinh_matran_pixel.py      # Script for camera calibration and pixel-to-world mapping
│   ├── control.py                # Core robotic motion profiling and trajectory control
│   └── chinhycodinhcontrol.py    # Optimized localized routine for stationary pick-and-place operations
├── ros2_ws/
│   └── src/                      # Native ROS2 workspaces, packages, and custom node configurations
├── homography_matrix.npy         # Saved spatial transformation matrix configuration
└── README.md                     # Project documentation
