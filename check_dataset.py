from pathlib import Path
import yaml

dataset_dir = Path("cattle-detection.v1i.yolov8")  # Название папки

# Читаем конфиг
with open(dataset_dir / "data.yaml") as f:
    config = yaml.safe_load(f)

print("📋 Конфиг датасета:")
print(f"   Классы: {config['names']}")
print(f"   Кол-во классов: {config['nc']}")

# Считаем файлы
for split in ["train", "valid", "test"]:
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    
    if img_dir.exists():
        n_images = len(list(img_dir.glob("*")))
        n_labels = len(list(lbl_dir.glob("*.txt")))
        print(f"\n📁 {split}:")
        print(f"   Картинок: {n_images}")
        print(f"   Разметок: {n_labels}")
        
        # Проверяем что совпадает
        if n_images != n_labels:
            print(f"НЕ СОВПАДАЕТ!")
        else:
            print(f"Всё ок")

# Смотрим пример разметки
example_label = next((dataset_dir / "train" / "labels").glob("*.txt"))
print(f"\n📝 Пример разметки ({example_label.name}):")
print(open(example_label).read())