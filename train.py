from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/cow_detection/my_first_train3/weights/best.pt")

    # 2. Запускаем дообучение
    results = model.train(
        data="data_new.yaml",   #конфиг с путями к папкам
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/cow_detection",
        name="tune_v1",
        
        lr0=0.001,              # Маленькая скорость обучения
        lrf=0.01,
        freeze=10,              
        
        
        workers=4               # Если опять упадет —  workers=0
    )

# ВОТ ЭТО ОБЯЗАТЕЛЬНО НУЖНО 
if __name__ == '__main__':
    main()