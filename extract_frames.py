import cv2
import os

# Настройки
VIDEO_PATH = 'testvid2.mp4'       # путь к вашему видео
OUTPUT_DIR = 'frames2'                # куда сохранять кадры
FRAME_INTERVAL = 30                  # каждый 30-й кадр (1 кадр в секунду при 30fps)

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % FRAME_INTERVAL == 0:
        filename = os.path.join(OUTPUT_DIR, f'frame_{saved:05d}.jpg')
        cv2.imwrite(filename, frame)
        saved += 1
    
    count += 1

cap.release()
print(f'Сохранено {saved} кадров в папку {OUTPUT_DIR}')