# visualize_labels.py
import cv2
import random
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

dataset_dir = Path("cattle-detection.v1i.yolov8")

with open(dataset_dir / "data.yaml") as f:
    config = yaml.safe_load(f)
class_names = config["names"]

images_dir = dataset_dir / "train" / "images"
labels_dir = dataset_dir / "train" / "labels"

# Берём 6 случайных картинок
all_images = list(images_dir.glob("*"))
samples = random.sample(all_images, min(6, len(all_images)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for ax, img_path in zip(axes.flat, samples):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Читаем разметку
    label_path = labels_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Из YOLO формата в пиксели
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Имя класса (может быть dict или list)
                if isinstance(class_names, dict):
                    name = class_names[cls_id]
                else:
                    name = class_names[cls_id]
                    
                cv2.putText(img, name, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    ax.imshow(img)
    ax.set_title(img_path.name, fontsize=8)
    ax.axis("off")

plt.suptitle("Проверка разметки датасета", fontsize=16)
plt.tight_layout()
plt.savefig("dataset_check.png", dpi=100)
plt.show()
print("✅ Сохранено в dataset_check.png")