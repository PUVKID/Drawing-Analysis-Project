from ultralytics import YOLO
model = YOLO("C:\\Users\\PUVKID\\Desktop\\심화캡스톤\\drawing_yolo_run\\v8_finetune\\weights\\best.pt")
model.predict("C:\\Users\\PUVKID\\Desktop\\심화캡스톤\\input\\1.png", save=True, conf=0.25)