#!/usr/bin/env python3
"""
V4: 6 类声音分类器
=================
- help_voice 由 ASR + 关键词匹配处理，不参与训练
- 6 类: scream, impact, help_knock, normal_speech, background, hard_negative
- Focal Loss + 早停
"""

import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── 配置 ──
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "classified_v2"
OUTPUT_DIR = Path(__file__).parent.parent / "model_output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 3
AUDIO_LEN = SAMPLE_RATE * DURATION
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 80
LR = 0.001
PATIENCE = 15

LABELS = ["scream", "impact", "help_knock", "normal_speech", "background", "hard_negative"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

# impact 类增强倍数（跌倒检测是核心功能）
IMPACT_AUG_TIMES = 3


class SoundDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx], self.labels[idx]

        try:
            audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        except Exception:
            audio = np.zeros(AUDIO_LEN, dtype=np.float32)

        if len(audio) > AUDIO_LEN:
            start = np.random.randint(0, len(audio) - AUDIO_LEN)
            audio = audio[start:start + AUDIO_LEN]
        else:
            audio = np.pad(audio, (0, max(0, AUDIO_LEN - len(audio))))

        if self.augment:
            audio = self._augment(audio, label)

        # MFCC + delta + delta2
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                     n_mels=N_MFCC, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        min_len = min(mfcc.shape[1], delta.shape[1], delta2.shape[1])
        features = np.stack([mfcc[:, :min_len], delta[:, :min_len], delta2[:, :min_len]])

        return torch.FloatTensor(features), label

    def _augment(self, audio, label):
        aug_prob = 0.8 if label == LABEL_MAP["impact"] else 0.5

        if np.random.random() < aug_prob:
            audio = audio * np.random.uniform(0.7, 1.3)

        if np.random.random() < aug_prob:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise

        if np.random.random() < aug_prob:
            rate = np.random.uniform(0.85, 1.15)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            if len(audio) > AUDIO_LEN:
                audio = audio[:AUDIO_LEN]
            else:
                audio = np.pad(audio, (0, max(0, AUDIO_LEN - len(audio))))

        if np.random.random() < aug_prob:
            t = np.random.randint(0, int(SAMPLE_RATE * 0.3))
            t0 = np.random.randint(0, max(1, len(audio) - t))
            audio[t0:t0 + t] = 0

        return audio.astype(np.float32)


class SoundCNN(nn.Module):
    """V1 成熟架构"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def load_dataset():
    files, labels = [], []
    class_counts = {i: 0 for i in range(len(LABELS))}

    for label_name in LABELS:
        class_dir = DATASET_DIR / label_name
        if not class_dir.exists():
            continue

        for f in class_dir.glob("*.wav"):
            files.append(f)
            labels.append(LABEL_MAP[label_name])
            class_counts[LABEL_MAP[label_name]] += 1

        # impact 类额外增强（跌倒检测是核心功能）
        if label_name == "impact":
            impact_files = list(class_dir.glob("*.wav"))
            for _ in range(IMPACT_AUG_TIMES):
                for f in impact_files:
                    files.append(f)
                    labels.append(LABEL_MAP["impact"])
                    class_counts[LABEL_MAP["impact"]] += 1

    print("类别分布:")
    for i, name in enumerate(LABELS):
        print(f"  {name}: {class_counts[i]}")

    return files, labels


def train():
    print("=" * 60)
    print("  声音分类 V3: 专注跌倒检测")
    print("=" * 60)

    files, labels = load_dataset()
    print(f"\n总样本: {len(files)}")

    # 打乱并划分
    indices = np.random.permutation(len(files))
    split = int(0.8 * len(files))
    train_idx, val_idx = indices[:split], indices[split:]

    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(f"训练集: {len(train_files)}, 验证集: {len(val_files)}")

    train_ds = SoundDataset(train_files, train_labels, augment=True)
    val_ds = SoundDataset(val_files, val_labels, augment=False)

    # 类别权重
    class_counts = np.bincount(train_labels, minlength=len(LABELS))
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(LABELS)
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SoundCNN(num_classes=len(LABELS)).to(device)
    print(f"设备: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=2.0, weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    patience_counter = 0
    best_fall_f1 = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        class_correct = np.zeros(len(LABELS))
        class_total = np.zeros(len(LABELS))
        fall_idx = LABEL_MAP["impact"]
        fall_fp = 0  # 非跌倒被误判为跌倒

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)

                for i in range(len(LABELS)):
                    mask = targets == i
                    class_total[i] += mask.sum().item()
                    class_correct[i] += predicted[mask].eq(i).sum().item()

                # 统计跌倒误报
                fall_mask = predicted == fall_idx
                fall_fp += (fall_mask & (targets != fall_idx)).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        dt = time.time() - t0
        scheduler.step()

        # 计算 impact F1
        fall_tp = class_correct[fall_idx]
        fall_precision = fall_tp / max(1, fall_tp + fall_fp)
        fall_recall = fall_tp / max(1, class_total[fall_idx])
        fall_f1 = 2 * fall_precision * fall_recall / max(0.001, fall_precision + fall_recall)

        # 打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{EPOCHS} ({dt:.1f}s)")
            print(f"  Train: loss={train_loss:.4f} acc={train_acc:.3f}")
            print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.3f}")
            print(f"  Impact F1: {fall_f1:.3f}")

            if (epoch + 1) % 10 == 0:
                for i, name in enumerate(LABELS):
                    if class_total[i] > 0:
                        acc = class_correct[i] / class_total[i]
                        print(f"    {name:12s}: {acc:.3f} ({int(class_correct[i])}/{int(class_total[i])})")

        # 保存最佳（综合考虑准确率和跌倒 F1）
        score = val_acc * 0.6 + fall_f1 * 0.4
        if score > best_acc:
            best_acc = score
            best_fall_f1 = fall_f1
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'labels': LABELS,
                'sample_rate': SAMPLE_RATE,
                'duration': DURATION,
                'n_mfcc': N_MFCC,
                'accuracy': val_acc,
                'fall_f1': fall_f1,
            }, str(OUTPUT_DIR / "best_model_v3.pth"))

            print(f"  ✓ 保存 (acc={val_acc:.3f}, fall_f1={fall_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停!")
                break

    # 最终
    print("\n" + "=" * 60)
    print("最终结果:")
    print(f"  最佳综合分: {best_acc:.3f}")
    print(f"  跌倒 F1: {best_fall_f1:.3f}")

    for i, name in enumerate(LABELS):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {name:12s}: {acc:.3f}")

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(LABELS, f)

    print(f"\n保存到: {OUTPUT_DIR / 'best_model_v3.pth'}")


if __name__ == "__main__":
    train()
