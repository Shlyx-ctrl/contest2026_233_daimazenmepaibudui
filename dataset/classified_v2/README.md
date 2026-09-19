# 声音分类数据集 V2 (6类)

数据集位置: 共享文件夹 D:\2026首届openvela AI硬件开发者大赛\dataset_v2

## 类别
- scream: 尖叫、哭喊声
- impact: 跌倒、撞击声
- help_knock: 敲门、敲桌、敲墙
- normal_speech: 普通聊天、咳嗽、笑声
- background: 安静、风扇、电视、走路、水声
- hard_negative: 关门、拍桌、掉书、移动椅子

## 数据量
| 类别 | 数量 |
|------|------|
| scream | 200 |
| impact | 280 (+840增强) |
| help_knock | 200 |
| normal_speech | 920 |
| background | 1000 |
| hard_negative | 281 |
| 总计 | 2881 |

## 使用方法
1. 将共享文件夹的 dataset_v2 复制到 dataset/classified_v2/
2. 运行: python3 tools/train_sound_v3.py
