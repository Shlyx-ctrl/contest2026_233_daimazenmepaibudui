#!/usr/bin/env python3
"""
从 FreeSound 收集跌倒/碰撞/摔倒声音
====================================
FreeSound API 需要免费 key：https://freesound.org/apiv2/apply/
即使没有 key，脚本也会尝试用公开的 preview 下载。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# 搜索关键词（英文效果最好）
QUERIES = [
    "body fall",
    "person falling down",
    "trip stumble fall",
    "heavy fall thud",
    "fall impact",
    "body hit ground",
    "collapse fall",
    "stumble fall",
    "slip and fall",
    "someone fell",
]

FALL_DIR = Path(__file__).parent.parent / "dataset" / "classified" / "fall"
PREVIEW_DIR = Path(__file__).parent.parent / "dataset" / "freesound_fall"


def search_freesound(query, api_key=None, page_size=50):
    """搜索 FreeSound"""
    params = {
        "query": query,
        "fields": "id,name,duration,samplerate,previews,download,tags,description",
        "page_size": min(page_size, 150),
        "sort": "rating_desc",
        "filter": "duration:[0.5 TO 15]",  # 0.5-15秒
    }
    if api_key:
        params["token"] = api_key

    url = f"https://freesound.org/apiv2/search/text/?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FallSoundCollector/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("results", [])
    except Exception as e:
        print(f"  ✗ 搜索失败 [{query}]: {e}")
        return []


def download_preview(item, dest_dir):
    """下载音频 preview"""
    previews = item.get("previews", {}) or {}
    preview_url = previews.get("preview-hq-mp3") or previews.get("preview-hq-ogg")
    if not preview_url:
        return False

    name = item.get("name", f"fs_{item['id']}")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80]
    ext = Path(preview_url).suffix or ".mp3"
    dest = dest_dir / f"{safe_name}{ext}"

    if dest.exists():
        return True

    try:
        req = urllib.request.Request(preview_url, headers={"User-Agent": "FallSoundCollector/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"    ⚠ 下载失败: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default="", help="FreeSound API key")
    parser.add_argument("--output", default=str(FALL_DIR), help="输出目录")
    parser.add_argument("--max-per-query", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output)
    preview_dir = Path(args.output).parent / "freesound_fall"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  从 FreeSound 收集跌倒/碰撞声音")
    print("=" * 50)

    total_downloaded = 0

    for query in QUERIES:
        print(f"\n🔍 搜索: {query}")
        results = search_freesound(query, args.api_key, args.max_per_query)
        if not results:
            continue

        downloaded = 0
        for i, item in enumerate(results):
            if download_preview(item, preview_dir):
                downloaded += 1
                total_downloaded += 1
            time.sleep(0.3)  # 限速

        print(f"  ✓ 下载 {downloaded}/{len(results)} 条")

    # 转换 mp3 → wav（用 ffmpeg 或直接复制）
    print(f"\n📂 转换为 WAV 格式...")
    import subprocess
    for mp3 in preview_dir.glob("*.mp3"):
        wav = output_dir / mp3.with_suffix(".wav").name
        if wav.exists():
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)],
                capture_output=True, timeout=10
            )
        except FileNotFoundError:
            # ffmpeg 不在，直接复制 mp3
            import shutil
            shutil.copy2(mp3, output_dir / mp3.name)

    print(f"\n✅ 完成！")
    print(f"  总计下载: {total_downloaded} 条")
    print(f"  输出目录: {output_dir}")

    # 统计
    count = len(list(output_dir.glob("*.*")))
    print(f"  fall 目录现有: {count} 个文件")


if __name__ == "__main__":
    main()
