'''
File Name: inference.py
Description: With this script we will do inference on real time camera feed, we will use yolov8 track mode for tracking objects.
'''
import cv2
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
# from google.colab.patches import cv2_imshow


def run(video_path, weight_file_path,count=0):
    # Load the YOLOv8 model
    # model = YOLO('/content/drive/MyDrive/datasets/runs/detect/train2/weights/best.pt')
    model = YOLO(weight_file_path)
    # d=0

    # Open the video file
    # video_path = "/content/drive/MyDrive/datasets/DashcamVdieo2_1_1.mp4"
    cap = cv2.VideoCapture(video_path)
    

    benchmark(model=weight_file_path, data='coco8.yaml', imgsz=640, half=False, device=0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow('detections',annotated_frame)
            filename = "file_%d.jpg"%d
            cv2.imwrite(filename, annotated_frame)
            count+=1
            # cv2.imwrite(output_path,annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()