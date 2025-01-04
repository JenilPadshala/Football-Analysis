# Learn how to use Ultralytics and YOLO and learn...
from ultralytics import YOLO

model = YOLO('yolov8x')
model.to('mps')
results = model.predict('input_videos/08fd33_4.mp4',device='mps',save=True)
print(results[0])
print("====================================")
for box in results[0].boxes:
    print(box)