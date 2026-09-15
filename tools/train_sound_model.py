#!/usr/bin/env python3
"""
声音分类模型训练 — 跌倒检测 & 环境声识别
==========================================
基于 ESC-50 数据集训练轻量级 CNN 模型，可导出 ONNX 用于嵌入式部署。

用法：
  # 训练模型（自动下载 ESC-50 如果没有）
  python3 train_sound_model.py

  # 自定义参数
  python3 train_sound_model.py --epochs 50 --batch-size 64 --lr 0.001

  # 仅评估已有模型
  python3 train_sound_model.py --eval-only --model best_model.pth

  # 导出 ONNX（用于嵌入式部署）
  python3 train_sound_model.py --export-onnx --model best_model.pth

依赖：
  pip install torch torchaudio librosa scikit-learn matplotlib onnx onnxruntime
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── 延迟导入（避免没装时报错信息友好）────────────────────────────

def check_deps():
    """检查依赖是否安装"""
    missing = []
    for pkg, imp in [("torch", "torch"), ("torchaudio", "torchaudio"),
                     ("librosa", "librosa"), ("sklearn", "sklearn"),
                     ("matplotlib", "matplotlib")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("✗ 缺少依赖，请安装：")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════════

# 标签映射（和 audio_dataset_scraper.py 对齐）
LABELS = ["cough", "door", "fall", "footsteps", "noise", "other", "scream", "water"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}

# 音频特征参数
SAMPLE_RATE = 16000       # 统一采样率
DURATION = 3.0            # 统一截取时长（秒）
N_MFCC = 40               # MFCC 特征维度
N_MELS = 64               # Mel 频谱维度
HOP_LENGTH = 512
N_FFT = 1024


# ══════════════════════════════════════════════════════════════════
#  数据集
# ══════════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    """
    从 classified/ 目录加载音频，提取 MFCC + Mel 频谱特征。
    """

    def __init__(self, data_dir, mode="mfcc", augment=False):
        """
        Args:
            data_dir: classified/ 目录路径
            mode: "mfcc" 或 "mel"
            augment: 是否做数据增强
        """
        self.mode = mode
        self.augment = augment
        self.samples = []  # [(file_path, label_idx)]

        data_dir = Path(data_dir)
        for label_dir in sorted(data_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            if label_name not in LABEL2IDX:
                continue
            label_idx = LABEL2IDX[label_name]
            for audio_file in label_dir.glob("*.*"):
                if audio_file.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac"):
                    self.samples.append((str(audio_file), label_idx))

        print(f"✓ 加载 {len(self.samples)} 个样本，{len(LABELS)} 个类别")

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        """加载并预处理音频"""
        try:
            # 用 librosa 加载（兼容性最好）
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            # 截取或填充到固定时长
            target_len = int(SAMPLE_RATE * DURATION)
            if len(audio) > target_len:
                # 随机裁剪（训练时）
                start = np.random.randint(0, len(audio) - target_len)
                audio = audio[start:start + target_len]
            else:
                # 零填充
                audio = np.pad(audio, (0, target_len - len(audio)))
            return audio.astype(np.float32)
        except Exception as e:
            print(f"  ⚠ 加载失败 {path}: {e}")
            return np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)

    def _extract_features(self, audio):
        """提取特征"""
        if self.mode == "mfcc":
            feat = librosa.feature.mfcc(
                y=audio, sr=SAMPLE_RATE,
                n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            # 加 delta
            delta = librosa.feature.delta(feat)
            delta2 = librosa.feature.delta(feat, order=2)
            feat = np.stack([feat, delta, delta2], axis=0)  # (3, n_mfcc, time)
        else:  # mel
            feat = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE,
                n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            feat = librosa.power_to_db(feat, ref=np.max)
            feat = feat[np.newaxis, ...]  # (1, n_mels, time)

        # 数据增强
        if self.augment:
            # 高斯噪声
            if np.random.random() < 0.3:
                feat += np.random.normal(0, 0.01, feat.shape).astype(np.float32)
            # 音量抖动
            if np.random.random() < 0.2:
                feat = feat * np.random.uniform(0.8, 1.2)
            # 时间遮蔽（模拟静音段）
            if np.random.random() < 0.2:
                t = feat.shape[2]
                mask_len = np.random.randint(1, max(2, t // 5))
                start = np.random.randint(0, max(1, t - mask_len))
                feat[:, :, start:start + mask_len] = 0
            # 频率遮蔽（模拟某些频段丢失）
            if np.random.random() < 0.2:
                f = feat.shape[1]
                mask_len = np.random.randint(1, max(2, f // 8))
                start = np.random.randint(0, max(1, f - mask_len))
                feat[:, start:start + mask_len, :] = 0

        return feat.astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = self._load_audio(path)
        feat = self._extract_features(audio)
        return torch.from_numpy(feat), label


# ══════════════════════════════════════════════════════════════════
#  模型：轻量级 CNN
# ══════════════════════════════════════════════════════════════════

class SoundCNN(nn.Module):
    """
    轻量级 CNN 用于环境声分类。

    输入: (batch, channels, freq, time)
      - MFCC 模式: (batch, 3, 40, time)
      - Mel 模式:  (batch, 1, 64, time)

    输出: (batch, num_classes)
    """

    def __init__(self, num_classes=8, in_channels=3, in_freq=40):
        super().__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 自适应池化 → 固定输出
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ══════════════════════════════════════════════════════════════════
#  训练
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * feats.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_labels)


# ══════════════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(labels, preds, output_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(LABELS))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(LABELS)

    # 标注数字
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ 混淆矩阵已保存: {path}")


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ 训练曲线已保存: {path}")


# ══════════════════════════════════════════════════════════════════
#  ONNX 导出
# ══════════════════════════════════════════════════════════════════

def export_onnx(model, output_dir, device):
    """导出 ONNX 模型（用于嵌入式部署）"""
    try:
        import onnx
    except ImportError:
        print("✗ 需要安装 onnx: pip install onnx")
        return

    model.eval()
    # MFCC 输入: (batch, 3, 40, 97) — 3秒@16kHz, hop=512 → ~97帧
    time_frames = int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1
    dummy = torch.randn(1, 3, N_MFCC, time_frames).to(device)

    onnx_path = output_dir / "sound_model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["mfcc"],
        output_names=["logits"],
        dynamic_axes={
            "mfcc": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=13,
    )

    # 验证
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print(f"✓ ONNX 模型已导出: {onnx_path}")
    print(f"  输入: mfcc {list(dummy.shape)}")
    print(f"  输出: logits [batch, {len(LABELS)}]")

    # 导出标签映射
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({"labels": LABELS, "idx2label": IDX2LABEL}, f, indent=2)
    print(f"✓ 标签映射已保存: {labels_path}")


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="声音分类模型训练")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="classified/ 目录路径（默认自动下载 ESC-50）")
    parser.add_argument("--output", type=str, default="model_output",
                        help="模型输出目录")
    parser.add_argument("--mode", choices=["mfcc", "mel"], default="mfcc",
                        help="特征模式: mfcc 或 mel")
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--model", type=str, default=None,
                        help="预训练模型路径（用于评估或继续训练）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估模型")
    parser.add_argument("--export-onnx", action="store_true",
                        help="导出 ONNX 模型")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: cuda / cpu（默认自动选择）")

    args = parser.parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  设备: {device}")

    # 数据目录
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        # 自动查找
        candidates = [
            Path("../dataset/classified"),
            Path("dataset/classified"),
            Path.home() / "contest_work/dataset/classified",
        ]
        for c in candidates:
            if c.exists():
                data_dir = c.resolve()
                break
        if data_dir is None:
            print("✗ 找不到数据目录，请用 --data-dir 指定 classified/ 路径")
            print("  或先运行: python3 audio_dataset_scraper.py --esc50")
            sys.exit(1)

    print(f"📂 数据目录: {data_dir}")

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print("\n📊 加载数据集...")
    in_freq = N_MFCC if args.mode == "mfcc" else N_MELS
    in_channels = 3 if args.mode == "mfcc" else 1

    dataset = AudioDataset(data_dir, mode=args.mode, augment=True)
    if len(dataset) == 0:
        print("✗ 没有找到音频文件")
        sys.exit(1)

    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # 验证集不做增强
    val_dataset.dataset.augment = False

    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # 模型
    print(f"\n🏗  构建模型: SoundCNN ({args.mode}, {in_channels}ch, {in_freq}freq)")
    model = SoundCNN(
        num_classes=len(LABELS),
        in_channels=in_channels,
        in_freq=in_freq
    ).to(device)

    if args.model and Path(args.model).exists():
        print(f"  加载预训练权重: {args.model}")
        model.load_state_dict(torch.load(args.model, map_location=device))

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 导出 ONNX
    if args.export_onnx:
        export_onnx(model, output_dir, device)
        return

    # 仅评估
    if args.eval_only:
        print("\n📊 评估模型...")
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.1f}%")
        print("\n分类报告:")
        print(classification_report(labels, preds, target_names=LABELS))
        plot_confusion_matrix(labels, preds, output_dir)
        return

    # 计算类别权重（处理不平衡）
    class_counts = [0] * len(LABELS)
    for _, label in dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = torch.tensor([total / (len(LABELS) * max(c, 1)) for c in class_counts])
    class_weights = class_weights / class_weights.sum() * len(LABELS)  # 归一化
    print(f"  类别权重: {dict(zip(LABELS, [f'{w:.2f}' for w in class_weights]))}")

    # 训练
    print(f"\n🚀 开始训练: {args.epochs} epochs, lr={args.lr}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        marker = " ★" if val_acc > best_acc else ""

        print(f"  Epoch {epoch:3d}/{args.epochs} │ "
              f"Train: {train_loss:.4f}/{train_acc:5.1f}% │ "
              f"Val: {val_loss:.4f}/{val_acc:5.1f}% │ "
              f"LR: {lr:.6f} │ {elapsed:.1f}s{marker}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            model_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)

    # 保存最终模型
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)

    print(f"\n✅ 训练完成！")
    print(f"  最佳验证准确率: {best_acc:.1f}%")
    print(f"  最佳模型: {output_dir / 'best_model.pth'}")
    print(f"  最终模型: {final_path}")

    # 绘图
    plot_training_curves(history, output_dir)

    # 最终评估
    print("\n📊 最终评估（最佳模型）:")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
    print(f"  Val Acc: {val_acc:.1f}%")
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=LABELS))
    plot_confusion_matrix(labels, preds, output_dir)

    # 保存训练信息
    info = {
        "labels": LABELS,
        "num_classes": len(LABELS),
        "mode": args.mode,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mfcc": N_MFCC if args.mode == "mfcc" else None,
        "n_mels": N_MELS if args.mode == "mel" else None,
        "best_accuracy": best_acc,
        "total_params": total_params,
        "train_samples": train_size,
        "val_samples": val_size,
    }
    info_path = output_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n✓ 模型信息已保存: {info_path}")


if __name__ == "__main__":
    main()
