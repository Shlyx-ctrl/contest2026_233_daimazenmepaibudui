# AI Coding 日志 - 2026-09-19

## 会话概要
- **成员**: Shlyx-ctrl
- **工具**: Claude Code (Claude Opus)
- **时长**: 约 5 小时
- **分支**: feat/sound-model-v4

## 完成工作

### 1. 模拟器编译与测试
- 修复 goldfish-arm64-v8a-ap 模拟器编译错误
- 创建板级 stub 头文件（sf32lb52_backlight.h, sf32lb52_alarm.h 等）
- 解决 LVGL v9 兼容性问题
- 成功在模拟器运行 robot_ui 应用

### 2. 技术报告
- 撰写完整技术报告（技术报告_智爱陪伴.md）
- 精简摘要至 293 字（<300 字限制）

### 3. 声音分类模型 V4
- **重新定义分类体系**: 6类（scream, impact, help_knock, normal_speech, background, hard_negative）
- **help_voice 由 ASR + 关键词匹配处理**
- 整合 ESC-50 数据集（2881 条训练样本）
- 训练模型：Impact F1 = 0.493，整体准确率 33.6%

### 4. 数据集管理
- 创建 classified_v2 目录结构
- 数据集存放在共享文件夹: D:\2026首届openvela AI硬件开发者大赛\dataset_v2
- 使用 Git LFS 管理大文件

### 5. Git 分支管理
- 创建 feat/sound-model-v4 分支
- 推送训练脚本和标签到 GitHub

## 关键文件变更
- tools/train_sound_v3.py（新增训练脚本）
- model_output/labels.json（6类标签）
- model_output/model_info.json（模型配置）
- dataset/classified_v2/README.md（数据集说明）

## 待办事项
- 成员一继续训练模型（运行 train_sound_v3.py）
- 用真实跌倒数据微调 impact 类
- 集成 ASR 关键词匹配逻辑
- 实机测试

## 技术决策记录
1. **分类体系**: 从 8 类改为 6 类，help_voice 由 ASR 处理
2. **数据增强**: impact 类做 3 倍增强（跌倒检测是核心功能）
3. **模型架构**: MFCC + CNN（SoundCNN），Focal Loss + 早停
4. **数据存储**: 大数据集放共享文件夹，模型和脚本放 GitHub
