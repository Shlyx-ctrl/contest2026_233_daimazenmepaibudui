# AI Coding 日志 — 2026-09-03

## 主要工作

### 1. LVGL Emoji 显示方案
- 研究 LVGL 显示 emoji 的方法（图标字体方案）
- 发现网络受限无法下载 FontAwesome / Material Icons
- 改用 ASCII 文字表情替代方案

### 2. robot_ui.c 表情更新
- 更新 face_array 中的 surprised 和 excited 表情
- 使用 ASCII art 风格的字符表情

## 使用的 AI 工具
- Claude Code (VS Code 插件)

## 涉及文件
- `app/robot_ui/robot_ui.c` — 表情数组更新
