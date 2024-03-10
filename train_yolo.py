'''
File Name: train_yolo.py
Description: With this script we can train yolov8 on our custom data and export the model in desired format. 
We are exporting it in Pytorch format.
'''
from ultralytics import YOLO
# from ultralytics.utils.benchmarks import benchmark

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model on out custom data .yaml file for 25 epochs. We can change the number of epochs as per requirement.
results = model.train(data='/content/drive/MyDrive/datasets/signage/data.yaml', epochs=25, imgsz=800, plots=True)

# benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to Pytorch format
# success = model.export(format='onnx')
success = model.export()