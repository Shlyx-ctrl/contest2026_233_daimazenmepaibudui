# AI Coding 日志 — 2026-09-18

## 主要工作

### 1. 接收队友音频数据
- 成员二录制了 WAV 音频文件（跌倒声、敲门声、尖叫声等）
- 通过 SCP 从 Windows 传输到 VM
- 解决中文路径编码问题（复制到 `D:\audio_temp` 再传输）

### 2. 数据集初步整理
- 分析音频格式（采样率、声道数、位深）
- 将音频按类别归入数据集目录
- 准备用于模型训练

### 3. 环境配置
- 在 VM 上安装 openssh-server（`sudo apt install openssh-server`）
- 配置 SSH 服务以支持文件传输

## 使用的 AI 工具
- Claude Code (VS Code 插件)

## 涉及文件
- `dataset/classified/` — 音频数据集目录
- WAV 音频文件
