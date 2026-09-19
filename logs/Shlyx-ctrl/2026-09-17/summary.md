# AI Coding 日志 — 2026-09-17

## 主要工作

### 1. 模型权重导出
- 编写 `export_model_c.py` 将 PyTorch 模型转为 C 头文件
- 导出 8 类声音分类模型权重为 C 数组
- 验证权重转换正确性

### 2. 模拟器测试
- 在 goldfish 模拟器上运行 robot_ui
- 发现 `robot_ui` 未注册为 NSH 内置命令
- 查找正确的 NuttX 应用启动方式

### 3. 编译问题修复
- 修复编译错误
- 确认模块在模拟器上可正常加载

## 使用的 AI 工具
- Claude Code (VS Code 插件)

## 涉及文件
- `tools/export_model_c.py`
- `app/robot_ui/CMakeLists.txt`
