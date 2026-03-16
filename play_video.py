import cv2
from ultralytics import YOLO

model_path = "runs/detect/runs/cow_detection/tune_v13/weights/best.pt"
model = YOLO(model_path)

# 2. Открываем видеофайл
video_path = "testvid1.mp4"  # Имя видеофайла
cap = cv2.VideoCapture(video_path)

# Проверка, открылось ли видео
if not cap.isOpened():
    print("Ошибка: Не могу открыть видеофайл!")
    exit()

print("Нажми 'Q' чтобы выйти")

while True:
    success, frame = cap.read()
    if not success:
        break # Видео кончилось

    # 3. Детекция
    results = model(frame, conf=0.25, imgsz=1280, device=0)

    # 4. Рисуем рамки
    annotated_frame = results[0].plot()

    # Добавляем счетчик коров на экран
    cow_count = len(results[0].boxes)
    cv2.putText(annotated_frame, f"Cows: {cow_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 5. Показываем 
    cv2.imshow("Cow Monitoring System", annotated_frame)

    # Выход по кнопке Q
    if cv2.waitKey(60) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()