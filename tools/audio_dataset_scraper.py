#!/usr/bin/env python3
"""
音频训练数据爬虫 — 跌倒检测 & 声音识别
========================================
从多个免费数据集源下载音频文件，用于训练跌倒检测和环境声分类模型。

数据源：
  1. ESC-50     — 50 类环境声（5 秒片段），直接下载压缩包
  2. FreeSound  — 通过 API 搜索并下载单条音频（需免费 API key）
  3. UrbanSound8K — 10 类城市声音（需手动下载后解压到本目录）

用法：
  # 1. 下载 ESC-50 数据集（最简单，一键搞定）
  python3 audio_dataset_scraper.py --esc50

  # 2. 从 FreeSound 搜索下载（需要 API key）
  python3 audio_dataset_scraper.py --freesound --api-key YOUR_KEY --keywords "fall,crash,glass breaking"

  # 3. 全部下载
  python3 audio_dataset_scraper.py --all

  # 4. 仅整理已有数据为训练目录结构
  python3 audio_dataset_scraper.py --organize

输出目录结构（整理后可直接喂给训练脚本）：
  dataset/
    fall/          — 跌倒、摔倒、碰撞
    glass/         — 玻璃碎裂
    scream/        — 尖叫、呼救
    cough/         — 咳嗽
    footsteps/     — 脚步声
    door/          — 开关门
    water/         — 流水、水龙头
    silence/       — 静音 / 背景噪声
    other/         — 其他
"""

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# ── ESC-50 类别映射（ESC-50 的 50 个类别 → 我们关心的标签）──────────────

ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "baby_cry", "clock_tick", "person_sneezing",
    "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "gun_shot", "fireworks", "hand_clapping", "door_wood_creaks", "can_opening",
    "water_drops", "wind", "pouring_water", "toilet_flush", "washing_machine",
    "clock_alarm", "glass_breaking", "dog_bark", "footsteps", "keyboard_typing",
    "mouse_click", "breathing", "cough", "snoring", "door_wood_knock",
    "church_bells", "siren", "baby_cry", "thunderstorm", "crackling_fire",
    "laughter", "brushing_teeth", "vacuum_cleaner", "typing", "sawing",
]

# 我们的标签 → ESC-50 类别名
ESC50_TARGET_MAP = {
    "fall":        ["glass_breaking", "chainsaw", "can_opening", "crackling_fire"],
    "glass":       ["glass_breaking"],
    "scream":      ["baby_cry", "person_sneezing", "dog_bark"],
    "cough":       ["cough", "breathing", "snoring"],
    "footsteps":   ["footsteps", "brushing_teeth", "keyboard_typing"],
    "door":        ["door_wood_creaks", "door_wood_knock", "can_opening"],
    "water":       ["water_drops", "pouring_water", "washing_machine"],
    "noise":       ["clock_tick", "wind", "rain", "sea_waves", "thunderstorm"],
    "other":       ["engine", "helicopter", "car_horn", "siren", "church_bells",
                    "laughter", "mouse_click", "typing", "sawing", "vacuum_cleaner",
                    "dog_bark", "sheep", "crow", "rain"],
}

# ESC-50 类别名 → 类别 ID（0-49）
ESC50_NAME_TO_ID = {name: i for i, name in enumerate(ESC50_LABELS)}

# ── FreeSound 搜索关键词映射 ──────────────────────────────────────

FREESOUND_KEYWORDS = {
    "fall":       ["falling", "body fall", "person falling", "trip stumble"],
    "glass":      ["glass break", "glass shatter", "window break"],
    "scream":     ["scream", "shout help", "woman scream", "yell"],
    "cough":      ["cough", "human cough", "coughing"],
    "footsteps":  ["footsteps", "walking", "person walking", "steps"],
    "door":       ["door slam", "door close", "door open", "door bang"],
    "water":      ["water running", "faucet", "water tap", "splash"],
    "noise":      ["background noise", "room tone", "ambient noise", "static"],
}


def progress_bar(current, total, width=40, prefix=""):
    """简单的进度条"""
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({pct:.0%})", end="", flush=True)
    if current == total:
        print()


def download_file(url, dest, desc=""):
    """下载文件，带重试"""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "AudioDatasetScraper/1.0 (research project)"
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            progress_bar(downloaded, total, prefix=desc)
            return True
        except Exception as e:
            if attempt < 2:
                print(f"\n  重试 ({attempt+1}/3): {e}")
                time.sleep(2)
            else:
                print(f"\n  下载失败: {e}")
                return False
    return False


# ══════════════════════════════════════════════════════════════════
#  ESC-50 数据集下载
# ══════════════════════════════════════════════════════════════════

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

def download_esc50(output_dir):
    """下载 ESC-50 并按我们的标签分类"""
    esc50_dir = output_dir / "raw" / "esc50"
    esc50_dir.mkdir(parents=True, exist_ok=True)

    zip_path = esc50_dir / "esc50.zip"

    if zip_path.exists():
        print(f"✓ ESC-50 压缩包已存在: {zip_path}")
    else:
        print("📥 下载 ESC-50 数据集 (~600MB)...")
        if not download_file(ESC50_URL, zip_path, "ESC-50"):
            print("✗ 下载失败")
            return

    # 解压
    extract_dir = esc50_dir / "ESC-50-master"
    if extract_dir.exists():
        print("✓ 已解压")
    else:
        print("📂 解压中...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(esc50_dir)
        print("✓ 解压完成")

    # 分类复制
    audio_dir = extract_dir / "audio"
    if not audio_dir.exists():
        # 尝试找实际路径
        for p in extract_dir.rglob("audio"):
            if p.is_dir():
                audio_dir = p
                break

    if not audio_dir.exists():
        print("✗ 找不到 audio 目录")
        return

    classified = output_dir / "classified"
    classified.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"📂 分类 {len(audio_files)} 个音频文件...")

    counts = {}
    for af in audio_files:
        # ESC-50 文件名格式: 1-100032-A-0.wav → 类别 ID 在第三个字段
        parts = af.stem.split("-")
        if len(parts) >= 4:
            try:
                label_id = int(parts[3])
            except ValueError:
                continue
            label_name = ESC50_LABELS[label_id] if label_id < len(ESC50_LABELS) else "unknown"

            # 找到属于哪个目标标签
            for target_label, esc_names in ESC50_TARGET_MAP.items():
                if label_name in esc_names:
                    dest_dir = classified / target_label
                    dest_dir.mkdir(exist_ok=True)
                    dest_file = dest_dir / af.name
                    if not dest_file.exists():
                        shutil.copy2(af, dest_file)
                    counts[target_label] = counts.get(target_label, 0) + 1
                    break

    print("\n✓ ESC-50 分类结果:")
    for label, count in sorted(counts.items()):
        print(f"  {label:12s}: {count:4d} 个音频")
    print(f"  {'合计':12s}: {sum(counts.values()):4d} 个音频")


# ══════════════════════════════════════════════════════════════════
#  FreeSound API 下载
# ══════════════════════════════════════════════════════════════════

def download_freesound(api_key, output_dir, keywords_map=None, max_per_query=20):
    """
    通过 FreeSound API 搜索并下载音频。

    申请免费 API key：https://freesound.org/apiv2/apply/
    """
    if keywords_map is None:
        keywords_map = FREESOUND_KEYWORDS

    classified = output_dir / "classified"
    classified.mkdir(parents=True, exist_ok=True)

    for label, keywords in keywords_map.items():
        print(f"\n🔍 搜索 [{label}]...")
        label_dir = classified / label
        label_dir.mkdir(exist_ok=True)

        for kw in keywords:
            # FreeSound API 搜索
            params = urllib.parse.urlencode({
                "query": kw,
                "token": api_key,
                "fields": "id,name,duration,previews,download",
                "page_size": min(max_per_query, 50),
                "sort": "rating_desc",
                "filter": "duration:[0 TO 30]",  # 30秒以内
            })
            url = f"https://freesound.org/apiv2/search/text/?{params}"

            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "AudioDatasetScraper/1.0"
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as e:
                print(f"  搜索失败 [{kw}]: {e}")
                time.sleep(2)
                continue

            results = data.get("results", [])
            print(f"  关键词「{kw}」找到 {len(results)} 条")

            for i, item in enumerate(results):
                name = item.get("name", f"unknown_{item['id']}")
                # 清理文件名
                safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
                safe_name = safe_name[:80]  # 截断过长的名字
                dest = label_dir / f"{safe_name}.wav"

                if dest.exists():
                    continue

                # 优先用 preview（小文件，免登录），fallback 到 download（需 auth）
                preview_url = (item.get("previews", {}) or {}).get("preview-hq-mp3")
                if not preview_url:
                    preview_url = (item.get("previews", {}) or {}).get("preview-hq-ogg")
                if not preview_url:
                    preview_url = item.get("download")

                if preview_url:
                    ext = Path(preview_url).suffix or ".mp3"
                    dest = dest.with_suffix(ext)
                    print(f"    [{i+1}/{len(results)}] {safe_name}{ext} ({item.get('duration', '?')}s)")
                    download_file(preview_url, dest, "")
                    time.sleep(0.5)  # 礼貌限速
                else:
                    print(f"    [{i+1}/{len(results)}] {safe_name} — 无下载链接")

            time.sleep(1)  # FreeSound API 限速


# ══════════════════════════════════════════════════════════════════
#  整理已有数据为训练目录
# ══════════════════════════════════════════════════════════════════

def organize_dataset(output_dir):
    """统计并展示最终的训练数据集结构"""
    classified = output_dir / "classified"
    if not classified.exists():
        print("✗ 没有找到 classified/ 目录，请先运行下载")
        return

    print("\n📊 训练数据集结构:")
    print("=" * 50)
    total = 0
    for label_dir in sorted(classified.iterdir()):
        if label_dir.is_dir():
            count = len(list(label_dir.glob("*.*")))
            total += count
            bar = "█" * min(count // 5, 30)
            print(f"  {label_dir.name:12s} │ {count:4d} │ {bar}")
    print("─" * 50)
    print(f"  {'合计':12s} │ {total:4d} │")

    # 生成文件列表（供训练脚本使用）
    file_list = output_dir / "file_list.txt"
    with open(file_list, "w") as f:
        for label_dir in sorted(classified.iterdir()):
            if label_dir.is_dir():
                for audio in sorted(label_dir.glob("*.*")):
                    f.write(f"{label_dir.name}\t{audio}\n")
    print(f"\n✓ 文件列表已保存: {file_list}")

    # 生成 CSV（常见格式）
    csv_file = output_dir / "labels.csv"
    with open(csv_file, "w") as f:
        f.write("filename,label\n")
        for label_dir in sorted(classified.iterdir()):
            if label_dir.is_dir():
                for audio in sorted(label_dir.glob("*.*")):
                    f.write(f"{audio},{label_dir.name}\n")
    print(f"✓ CSV 标签已保存: {csv_file}")


# ══════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="音频训练数据爬虫 — 跌倒检测 & 声音识别"
    )
    parser.add_argument("--esc50", action="store_true",
                        help="下载 ESC-50 环境声数据集（~600MB）")
    parser.add_argument("--freesound", action="store_true",
                        help="从 FreeSound 搜索下载（需 API key）")
    parser.add_argument("--api-key", type=str, default="",
                        help="FreeSound API key")
    parser.add_argument("--keywords", type=str, default="",
                        help="FreeSound 额外搜索关键词（逗号分隔）")
    parser.add_argument("--max-per-query", type=int, default=20,
                        help="每个关键词最多下载几条（默认 20）")
    parser.add_argument("--all", action="store_true",
                        help="下载所有数据源")
    parser.add_argument("--organize", action="store_true",
                        help="整理并展示数据集结构")
    parser.add_argument("--output", type=str, default="dataset",
                        help="输出目录（默认 dataset/）")

    args = parser.parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  音频训练数据爬虫 — 跌倒检测 & 声音识别")
    print("=" * 50)
    print(f"  输出目录: {output_dir}\n")

    if not any([args.esc50, args.freesound, args.all, args.organize]):
        parser.print_help()
        print("\n💡 快速开始：先下载 ESC-50 数据集试试")
        print("   python3 audio_dataset_scraper.py --esc50")
        return

    if args.esc50 or args.all:
        download_esc50(output_dir)

    if args.freesound or args.all:
        if not args.api_key:
            print("\n⚠️  FreeSound 需要 API key")
            print("   申请地址: https://freesound.org/apiv2/apply/")
            print("   用法: python3 audio_dataset_scraper.py --freesound --api-key YOUR_KEY")
        else:
            extra_kw = {}
            if args.keywords:
                for kw in args.keywords.split(","):
                    kw = kw.strip()
                    if kw:
                        extra_kw.setdefault("custom", []).append(kw)
            kw_map = dict(FREESOUND_KEYWORDS)
            if extra_kw.get("custom"):
                kw_map["custom"] = extra_kw["custom"]
            download_freesound(args.api_key, output_dir, kw_map, args.max_per_query)

    if args.organize or args.all:
        organize_dataset(output_dir)


if __name__ == "__main__":
    main()
