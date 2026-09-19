# AI Coding 日志 — 2026-09-15

## 主要工作

### 1. 声音数据采集（FreeSound 爬虫）
- 编写 `collect_fall_sounds.py` 从 FreeSound 爬取跌倒声数据
- 自动下载并分类音频文件

### 2. 训练脚本优化
- 数据增强：3 倍增强 impact 类别
- 超参数调优：学习率、batch size、early stopping
- 修复 fall F1 计算性能 bug（从 376 次推理降至验证循环内计算）

### 3. 板端推理代码
- 编写嵌入式推理代码
- 模型权重转 C 数组
- MFCC 特征提取的 C 实现

## 使用的 AI 工具
- Claude Code (VS Code 插件)

## 涉及文件
- `tools/collect_fall_sounds.py`
- `tools/train_sound_v3.py`
- `app/robot_ui/sound_classifier.c`
- `app/robot_ui/model_weights.c`
