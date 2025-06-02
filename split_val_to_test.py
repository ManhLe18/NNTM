import os
import shutil
import random

val_dir = r'C:\Users\ADMIN\Downloads\archive (1)\RiceDiseaseDataset\validation'
test_dir = r'C:\Users\ADMIN\Downloads\archive (1)\RiceDiseaseDataset\test'
target_test_images = 92  # Số lượng ảnh muốn chuyển sang test

os.makedirs(test_dir, exist_ok=True)

# Đếm tổng số ảnh trong validation
total_images = 0
class_image_counts = {}
for class_name in os.listdir(val_dir):
    class_val_dir = os.path.join(val_dir, class_name)
    if not os.path.isdir(class_val_dir):
        continue
    images = [f for f in os.listdir(class_val_dir) if os.path.isfile(os.path.join(class_val_dir, f))]
    class_image_counts[class_name] = len(images)
    total_images += len(images)

# Tính tỷ lệ chuyển cho mỗi class dựa trên số lượng ảnh
for class_name in os.listdir(val_dir):
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_val_dir):
        continue
    
    os.makedirs(class_test_dir, exist_ok=True)
    images = [f for f in os.listdir(class_val_dir) if os.path.isfile(os.path.join(class_val_dir, f))]
    random.shuffle(images)
    
    # Tính số ảnh cần chuyển cho class này
    n_test = int((class_image_counts[class_name] / total_images) * target_test_images)
    test_images = images[:n_test]
    
    for img in test_images:
        src = os.path.join(class_val_dir, img)
        dst = os.path.join(class_test_dir, img)
        shutil.move(src, dst)

print('Tach du lieu xong!') 