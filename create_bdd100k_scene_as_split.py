"""
scene(장면 유형)을 split으로 사용하는 weather 데이터셋 생성.

  train <- city street
  val   <- highway
  test  <- 나머지 전부 (residential / parking lot / gas stations / tunnel / undefined 등)

각 줄 포맷: "{image_path} {weather_label}"  (region_as_split과 동일)
BDD100K 라벨 JSON(attributes.scene, attributes.weather)을 직접 읽는다.
(region_as_split과 달리 GPS 지역 분류가 필요 없어 사전 계산된 txt 없이 바로 만든다.)
"""
import re
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# ── paths ─────────────────────────────────────────────────────────────────────
BDD100K_BASE = "/home/young/AILabDataset/01_Open_Dataset/26_BDD-100K"
LABEL_BASE   = os.path.join(BDD100K_BASE, "labels/100k")  # per-image 라벨 JSON
IMAGE_BASE   = os.path.join(BDD100K_BASE, "images/100k")
OFFICIAL_SPLITS = ("train", "val", "test")  # BDD100K 공식 split (label 폴더 3개 모두 존재)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "data", "scene_as_split_no_foggy")
PLOT_DIR = os.path.join(SAVE_DIR, "plots")

# ── label mapping (BDD100K 전체 weather 클래스, foggy 포함) ───────────────────────
# clear/partly cloudy 는 동일 의미라 0으로 통합. undefined 는 실제 날씨가 아니므로 스킵.
# WEATHER_MAP = {
#     "clear":         0,
#     "partly cloudy": 0,
#     "overcast":      1,
#     "foggy":         2,
#     "rainy":         3,
#     "snowy":         4,
# }
WEATHER_MAP = {
    "clear":         0,
    "partly cloudy": 0,
    "overcast":      1,
    "rainy":         2,
    "snowy":         3,
}
# WEATHER_CLASS_NAMES = {0: "Clear", 1: "Overcast", 2: "Foggy", 3: "Rainy", 4: "Snowy"}
# WEATHER_COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

WEATHER_CLASS_NAMES = {0: "Clear", 1: "Overcast", 2: "Rainy", 3: "Snowy"}
WEATHER_COLORS      = ["#4C72B0", "#DD8452", "#C44E52", "#8172B2"]

# scene -> 목표 split. 여기 없는 scene(residential/parking lot/gas stations/tunnel/undefined 등)은 전부 test.
SCENE_TO_SPLIT = {
    "city street": "train",
    "highway":     "val",
}
DEFAULT_SPLIT = "test"
SPLIT_SCENE_NAME = {
    "train": "city street",
    "val":   "highway",
    "test":  "Other (residential/parking lot/gas stations/tunnel/undefined, etc.)",
}

WEATHER_RE = re.compile(r'"weather"\s*:\s*"([^"]+)"')
SCENE_RE   = re.compile(r'"scene"\s*:\s*"([^"]+)"')


def iter_labels():
    """BDD100K 라벨 JSON 전체를 순회하며 (image_path, scene, weather) yield."""
    for off_split in OFFICIAL_SPLITS:
        split_dir = os.path.join(LABEL_BASE, off_split)
        fnames = sorted(f for f in os.listdir(split_dir) if f.endswith(".json"))
        for fname in tqdm(fnames, desc=f"scene/weather 라벨링 ({off_split})"):
            with open(os.path.join(split_dir, fname)) as f:
                text = f.read()
            m_scene = SCENE_RE.search(text)
            m_weather = WEATHER_RE.search(text)
            scene = m_scene.group(1) if m_scene else "undefined"
            weather = m_weather.group(1) if m_weather else None

            key = os.path.splitext(fname)[0]
            img_path = os.path.join(IMAGE_BASE, off_split, key + ".jpg")
            yield img_path, scene, weather


def build():
    """scene=split, 라벨=weather 로 train/val/test 데이터를 구성한다."""
    paths = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    n_no_label = n_skip_weather = 0

    for img_path, scene, weather in iter_labels():
        if weather is None:
            n_no_label += 1
            continue
        if weather not in WEATHER_MAP:          # undefined 등 매핑 외 값 스킵
            n_skip_weather += 1
            continue

        tgt = SCENE_TO_SPLIT.get(scene, DEFAULT_SPLIT)
        paths[tgt].append(img_path)
        labels[tgt].append(WEATHER_MAP[weather])

    print(f"\n  라벨 파일 없음: {n_no_label}, weather 스킵(undefined): {n_skip_weather}")
    out_p = {s: np.array(paths[s]) for s in paths}
    out_l = {s: np.array(labels[s], dtype=np.int32) for s in labels}
    return out_p, out_l


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_txt(image_paths, labels, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        for img, lbl in zip(image_paths, labels):
            f.write(f"{img} {lbl}\n")
    print(f"  Saved {len(image_paths):,} entries -> {dst}")


# ── reporting / plotting ────────────────────────────────────────────────────────

def print_distribution(labels, split_name, class_names):
    total = len(labels)
    scene = SPLIT_SCENE_NAME[split_name]
    print(f"\n[{split_name} = {scene}] Weather distribution (total: {total:,})")
    for cls_id, cls_name in class_names.items():
        count = int(np.sum(labels == cls_id))
        pct   = count / total * 100 if total > 0 else 0
        print(f"  class {cls_id} ({cls_name}): {count:>7,}  ({pct:.1f}%)")


def plot_distribution(labels, title, save_path, class_names, colors):
    n_classes = max(class_names.keys()) + 1
    counts    = [int(np.sum(labels == i)) for i in range(n_classes)]
    total     = sum(counts)

    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [class_names[i] for i in range(n_classes)],
        counts,
        color=colors[:n_classes], edgecolor="white", linewidth=0.8,
    )
    for bar, count in zip(bars, counts):
        pct = count / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.005,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Weather", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    paths, labels = build()

    for sp in ("train", "val", "test"):
        print_distribution(labels[sp], sp, WEATHER_CLASS_NAMES)

    print()
    for sp in ("train", "val", "test"):
        write_txt(paths[sp], labels[sp], os.path.join(SAVE_DIR, f"{sp}.txt"))

    print()
    for sp in ("train", "val", "test"):
        scene = SPLIT_SCENE_NAME[sp]
        plot_distribution(
            labels[sp],
            title=f"BDD100K Weather — {sp} ({scene}) (No Foggy)",
            save_path=os.path.join(PLOT_DIR, f"weather_{sp}.png"),
            class_names=WEATHER_CLASS_NAMES,
            colors=WEATHER_COLORS,
        )

    print("\nDone.")
