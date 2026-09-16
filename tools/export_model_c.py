#!/usr/bin/env python3
"""
将 PyTorch 模型权重导出为 C 头文件
===================================
用于嵌入式部署，直接编译进固件
"""

import torch
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model_output" / "best_model.pth"
OUTPUT_PATH = Path(__file__).parent.parent / "app" / "robot_ui" / "sound_model_weights.h"


def fuse_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """融合 Conv + BatchNorm"""
    # BN: y = (x - mean) / sqrt(var + eps) * weight + bias
    # Conv: y = conv(x) + bias
    # 融合后: weight_fused = weight * bn_weight / sqrt(var + eps)
    #         bias_fused = (conv_bias - mean) * bn_weight / sqrt(var + eps) + bn_bias

    scale = bn_weight / np.sqrt(bn_var + eps)
    weight_fused = conv_weight * scale.reshape(-1, 1, 1, 1)
    bias_fused = (conv_bias - bn_mean) * scale + bn_bias

    return weight_fused, bias_fused


def flatten_weight(w):
    """将权重展平为一维数组"""
    return w.flatten()


def format_array(name, data, dtype="float"):
    """格式化为 C 数组"""
    lines = []
    lines.append(f"static const {dtype} {name}[] = {{")

    # 每行 8 个数
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        values = ", ".join(f"{v:.8f}f" for v in chunk)
        lines.append(f"    {values},")

    lines.append("};")
    return "\n".join(lines)


def main():
    print("加载模型...")
    model_data = torch.load(MODEL_PATH, map_location="cpu")

    # 提取参数
    conv1_w = model_data["conv1.weight"].numpy()
    conv1_b = model_data["conv1.bias"].numpy()
    bn1_w = model_data["bn1.weight"].numpy()
    bn1_b = model_data["bn1.bias"].numpy()
    bn1_mean = model_data["bn1.running_mean"].numpy()
    bn1_var = model_data["bn1.running_var"].numpy()

    conv2_w = model_data["conv2.weight"].numpy()
    conv2_b = model_data["conv2.bias"].numpy()
    bn2_w = model_data["bn2.weight"].numpy()
    bn2_b = model_data["bn2.bias"].numpy()
    bn2_mean = model_data["bn2.running_mean"].numpy()
    bn2_var = model_data["bn2.running_var"].numpy()

    conv3_w = model_data["conv3.weight"].numpy()
    conv3_b = model_data["conv3.bias"].numpy()
    bn3_w = model_data["bn3.weight"].numpy()
    bn3_b = model_data["bn3.bias"].numpy()
    bn3_mean = model_data["bn3.running_mean"].numpy()
    bn3_var = model_data["bn3.running_var"].numpy()

    fc1_w = model_data["fc1.weight"].numpy()
    fc1_b = model_data["fc1.bias"].numpy()
    fc2_w = model_data["fc2.weight"].numpy()
    fc2_b = model_data["fc2.bias"].numpy()

    # 融合 BatchNorm
    print("融合 BatchNorm...")
    conv1_w_fused, conv1_b_fused = fuse_bn(conv1_w, conv1_b, bn1_w, bn1_b, bn1_mean, bn1_var)
    conv2_w_fused, conv2_b_fused = fuse_bn(conv2_w, conv2_b, bn2_w, bn2_b, bn2_mean, bn2_var)
    conv3_w_fused, conv3_b_fused = fuse_bn(conv3_w, conv3_b, bn3_w, bn3_b, bn3_mean, bn3_var)

    # 生成 C 头文件
    print("生成 C 头文件...")
    c_code = []

    c_code.append("/*")
    c_code.append(" * sound_model_weights.h - 声音分类模型权重")
    c_code.append(" * ")
    c_code.append(" * 自动生成，请勿手动编辑")
    c_code.append(" * 模型: SoundCNN (3层Conv + 2层FC)")
    c_code.append(" * 准确率: 60.5%")
    c_code.append(" * 标签: cough, door, fall, footsteps, noise, other, scream, water")
    c_code.append(" */")
    c_code.append("")
    c_code.append("#ifndef SOUND_MODEL_WEIGHTS_H")
    c_code.append("#define SOUND_MODEL_WEIGHTS_H")
    c_code.append("")
    c_code.append("#include <stdint.h>")
    c_code.append("")
    c_code.append("/* 模型参数 */")
    c_code.append("#define SOUND_MODEL_CONV1_IN_CHANNELS   3")
    c_code.append("#define SOUND_MODEL_CONV1_OUT_CHANNELS  32")
    c_code.append("#define SOUND_MODEL_CONV2_IN_CHANNELS   32")
    c_code.append("#define SOUND_MODEL_CONV2_OUT_CHANNELS  64")
    c_code.append("#define SOUND_MODEL_CONV3_IN_CHANNELS   64")
    c_code.append("#define SOUND_MODEL_CONV3_OUT_CHANNELS 128")
    c_code.append("#define SOUND_MODEL_FC1_IN_FEATURES    2048")
    c_code.append("#define SOUND_MODEL_FC1_OUT_FEATURES    256")
    c_code.append("#define SOUND_MODEL_FC2_IN_FEATURES     256")
    c_code.append("#define SOUND_MODEL_FC2_OUT_FEATURES      8")
    c_code.append("")

    # Conv1 (已融合 BN)
    c_code.append("/* Conv1 + BN1 (已融合) */")
    c_code.append(format_array("sound_model_conv1_weight", flatten_weight(conv1_w_fused)))
    c_code.append(format_array("sound_model_conv1_bias", conv1_b_fused))
    c_code.append("")

    # Conv2 (已融合 BN)
    c_code.append("/* Conv2 + BN2 (已融合) */")
    c_code.append(format_array("sound_model_conv2_weight", flatten_weight(conv2_w_fused)))
    c_code.append(format_array("sound_model_conv2_bias", conv2_b_fused))
    c_code.append("")

    # Conv3 (已融合 BN)
    c_code.append("/* Conv3 + BN3 (已融合) */")
    c_code.append(format_array("sound_model_conv3_weight", flatten_weight(conv3_w_fused)))
    c_code.append(format_array("sound_model_conv3_bias", conv3_b_fused))
    c_code.append("")

    # FC1
    c_code.append("/* FC1 */")
    c_code.append(format_array("sound_model_fc1_weight", flatten_weight(fc1_w)))
    c_code.append(format_array("sound_model_fc1_bias", fc1_b))
    c_code.append("")

    # FC2
    c_code.append("/* FC2 */")
    c_code.append(format_array("sound_model_fc2_weight", flatten_weight(fc2_w)))
    c_code.append(format_array("sound_model_fc2_bias", fc2_b))
    c_code.append("")

    c_code.append("#endif /* SOUND_MODEL_WEIGHTS_H */")

    # 写入文件
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(c_code))

    # 统计大小
    total_params = (
        conv1_w_fused.size + conv1_b_fused.size +
        conv2_w_fused.size + conv2_b_fused.size +
        conv3_w_fused.size + conv3_b_fused.size +
        fc1_w.size + fc1_b.size +
        fc2_w.size + fc2_b.size
    )
    file_size = OUTPUT_PATH.stat().st_size

    print(f"\n✅ 导出成功!")
    print(f"  文件: {OUTPUT_PATH}")
    print(f"  大小: {file_size / 1024:.1f} KB")
    print(f"  参数量: {total_params:,}")
    print(f"  内存占用: {total_params * 4 / 1024:.1f} KB (float32)")


if __name__ == "__main__":
    main()
