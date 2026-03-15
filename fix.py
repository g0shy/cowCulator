from ultralytics import YOLO

# 1. Загружаем твою модель (проверь путь!)
path = "runs/detect/runs/cow_detection/my_first_train3/weights/best.pt"
model = YOLO(path)

# 2. Новые имена (все одинаковые)
new_names = {
    0: 'Cow',
    1: 'Cow',
    2: 'Cow'
}

# === ИСПРАВЛЕНИЕ ===
# Меняем ТОЛЬКО внутри самой нейросети
model.model.names = new_names

# 3. Сохраняем "Чистую" модель
model.save("cow_clean.pt")

print("✅ Готово! Файл cow_clean.pt создан.")
print(f"Проверка имен внутри нового файла: {model.model.names}")