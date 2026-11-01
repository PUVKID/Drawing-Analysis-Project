from ultralytics import YOLO
model = YOLO('./drawing_yolo_run/v8_finetune/weights/best.pt')
model.predict('./input/1.png', save=True, conf=0.25)