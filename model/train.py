import os
import sys
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from collections import Counter
from tqdm import tqdm
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from albumentations.pytorch import ToTensorV2
import albumentations as A
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🟢 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AlbumentationsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = []
        self.labels = []
        self.classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
        self.transform = transform
        for cls in self.classes:
            class_dir = os.path.join(root, cls)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, fname))
                        self.labels.append(self.classes.index(cls))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

def tta_predict(model, img, tta_transforms):
    predictions = []
    with torch.no_grad():
        # Original image
        pred = model(img.unsqueeze(0))
        predictions.append(pred)
        
        # TTA predictions
        for transform in tta_transforms:
            aug_img = transform(image=img.cpu().numpy().transpose(1, 2, 0))['image']
            aug_img = torch.from_numpy(aug_img.transpose(2, 0, 1)).float().to(device)
            pred = model(aug_img.unsqueeze(0))
            predictions.append(pred)
    
    # Average predictions
    return torch.mean(torch.stack(predictions), dim=0)

def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    # TTA transforms
    tta_transforms = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0)
    ]
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Use TTA for prediction
            outputs = tta_predict(model, imgs[0], tta_transforms)
            for i in range(1, len(imgs)):
                output = tta_predict(model, imgs[i], tta_transforms)
                outputs = torch.cat((outputs, output), dim=0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Bảng chi tiết như mẫu
    print("\nBảng 2. Chi tiết độ chính xác trên từng lớp")
    print(f"{'Tên bệnh':<20}{'Số ảnh nhận dạng đúng':<20}{'Số ảnh nhận dạng sai':<20}{'Độ chính xác (%)':<20}")
    total_acc = 0
    n_class = len(class_names)
    
    # Tạo DataFrame cho bảng chi tiết
    detail_data = []
    for i, class_name in enumerate(class_names):
        correct = conf_matrix[i][i]
        total = conf_matrix[i].sum()
        wrong = total - correct
        acc = 100 * correct / total if total > 0 else 0
        total_acc += acc
        print(f"{class_name:<20}{correct:<20}{wrong:<20}{acc:<20.1f}")
        detail_data.append({
            'Tên bệnh': class_name,
            'Số ảnh nhận dạng đúng': correct,
            'Số ảnh nhận dạng sai': wrong,
            'Độ chính xác (%)': round(acc, 2)
        })
    
    avg_acc = total_acc / n_class
    print(f"{'Trung bình':<60}{avg_acc:<20.1f}")
    detail_data.append({
        'Tên bệnh': 'Trung bình',
        'Số ảnh nhận dạng đúng': '',
        'Số ảnh nhận dạng sai': '',
        'Độ chính xác (%)': round(avg_acc, 2)
    })
    
    # In các chỉ số đánh giá
    print("\nBảng 3. Các chỉ số đánh giá trên từng lớp (precision, recall, f1-score, support):")
    print(report)
    
    # In confusion matrix dạng số
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    try:
        # Xuất ra Excel
        with pd.ExcelWriter('model_evaluation.xlsx', engine='openpyxl') as writer:
            # Sheet 1: Chi tiết độ chính xác
            pd.DataFrame(detail_data).to_excel(writer, sheet_name='Chi tiết độ chính xác', index=False)
            
            # Sheet 2: Các chỉ số đánh giá
            metrics_data = []
            for class_name in class_names:
                metrics_data.append({
                    'Lớp': class_name,
                    'Precision': round(report_dict[class_name]['precision'], 3),
                    'Recall': round(report_dict[class_name]['recall'], 3),
                    'F1-score': round(report_dict[class_name]['f1-score'], 3),
                    'Support': report_dict[class_name]['support']
                })
            metrics_data.append({
                'Lớp': 'accuracy',
                'Precision': round(report_dict['accuracy'], 3),
                'Recall': '',
                'F1-score': '',
                'Support': report_dict['macro avg']['support']
            })
            metrics_data.append({
                'Lớp': 'macro avg',
                'Precision': round(report_dict['macro avg']['precision'], 3),
                'Recall': round(report_dict['macro avg']['recall'], 3),
                'F1-score': round(report_dict['macro avg']['f1-score'], 3),
                'Support': report_dict['macro avg']['support']
            })
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Các chỉ số đánh giá', index=False)
            
            # Sheet 3: Confusion Matrix
            conf_matrix_df = pd.DataFrame(conf_matrix, 
                                        index=class_names,
                                        columns=class_names)
            conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
            
            # Sheet 4: Thông tin model
            model_info = {
                'Thông tin': [
                    'Ngày đánh giá',
                    'Số lớp',
                    'Tên các lớp',
                    'Tổng số ảnh validation',
                    'Độ chính xác trung bình'
                ],
                'Giá trị': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(class_names),
                    ', '.join(class_names),
                    len(all_labels),
                    round(avg_acc, 2)
                ]
            }
            pd.DataFrame(model_info).to_excel(writer, sheet_name='Thông tin model', index=False)
        
        print("\n✅ Đã xuất kết quả đánh giá ra file 'model_evaluation.xlsx'")
    except Exception as e:
        print(f"\n❌ Lỗi khi xuất Excel: {str(e)}")
    
    return conf_matrix, report_dict

def train_model():
    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    freeze_epochs = 2  # unfreeze sớm hơn
    lr = 1e-3
    weight_decay = 1e-4
    num_classes = 4
    patience = 10
    mixup_alpha = 0.2  # Mixup parameter

    # Paths
    train_dir = r"C:\Users\ADMIN\Downloads\archive (1)\RiceDiseaseDataset\train"
    val_dir = r"C:\Users\ADMIN\Downloads\archive (1)\RiceDiseaseDataset\validation"

    # Transforms (giảm augmentation mạnh)
    train_tf = A.Compose([
        A.Resize(380, 380),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Dataset & Loader
    train_ds = AlbumentationsDataset(train_dir, transform=train_tf)
    val_ds = AlbumentationsDataset(val_dir, transform=val_tf)

    class_counts = Counter(train_ds.labels)
    sample_weights = [1.0 / class_counts[l] for l in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    # Loss & optimizer (no label smoothing)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_acc = 0.0
    early_stop_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    print("🚀 Start training...")
    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch+1}/{num_epochs}")

        # Freeze
        if epoch == 0:
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
            print("🔒 Freezing backbone layers.")
        elif epoch == freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            print("🔓 Unfreezing entire model.")

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Apply Mixup
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + len(train_loader))
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (lam * (preds == labels_a).sum().float() + 
                       (1 - lam) * (preds == labels_b).sum().float())
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100 * correct_val / total_val

        print(f"✅ Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, "best_model.pth")
            print("💾 Saved best model")
        else:
            early_stop_counter += 1
            if early_stop_counter > patience:
                print("⏹️ Early stopping due to no improvement")
                break

    model.load_state_dict(best_weights)
    
    # Evaluate final model
    print("\n🔍 Evaluating final model...")
    evaluate_model(model, val_loader, train_ds.classes)
    
    return model

if __name__ == "__main__":
    train_model()