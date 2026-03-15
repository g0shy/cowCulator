from ultralytics import YOLO


def main():
    # Загружаем модель
    model = YOLO("yolo11m.pt")

    # Запускаем обучение
    results = model.train(
        
        data="/home/zer0p1e/my_yolo_project/cattle-detection.v1i.yolov8/data.yaml",
        
        epochs=100,
        imgsz=640,
        device=0,      
        batch=16,      
        
        project="runs/cow_detection",
        name="my_first_train",
        workers=0   
        
    )


if __name__ == '__main__':
    main()