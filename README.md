(A) YOLO Intrusion Detection System:
An AI-powered real-time intrusion detection system that uses YOLOv8 (You Only Look Once) for object detection to identify human presence in restricted areas.
When a person is detected within a defined region, the system triggers an alert and captures evidence automatically.

(B) Features:
1. Real-time object detection using YOLOv8
2. Intrusion alert system (visual + audio beep)
3. Automatic frame saving of intrusion events
4. Tracks only human (person class) detections
5. Adjustable restricted zones
6. Generates annotated output videos

(C)Tech Stacks used: 
Component	        Technology
Object Detection	YOLOv8 (Ultralytics)
Programming Language	Python
Libraries Used  	OpenCV, NumPy, os, time, sys
Model File	        yolov8n.pt
Environment	        Local Machine / Jupyter / VS Code

