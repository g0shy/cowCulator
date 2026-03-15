import cv2
from ultralytics import YOLO

model = YOLO("cow_clean.pt")

# Открываем видеофайл
video_path = "testvid1.mp4"  # Имя идеофайла
cap = cv2.VideoCapture(video_path)

# Проверка
if not cap.isOpened():
    print("Ошибка: Не могу открыть видеофайл!")
    exit()

print("Нажми 'Q' чтобы выйти")

while True:
    success, frame = cap.read()
    if not success:
        break # Видео кончилось

    # 3. Детекция
    # conf=0.5 — показывать только если уверенность > 50%
    # results = model(frame, conf=0.5, device=0)
    results = model(frame, conf=0.25, imgsz=1280, device=0)
    
    # рамки
    annotated_frame = results[0].plot()

    # Добавляем счетчик коров на экран
    cow_count = len(results[0].boxes)
    cv2.putText(annotated_frame, f"Cows: {cow_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 5. Показываем (уменьшим окно, если видео 4K)
    cv2.imshow("Cow Monitoring System", annotated_frame)

    # Выход по кнопке Q
    if cv2.waitKey(60) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()