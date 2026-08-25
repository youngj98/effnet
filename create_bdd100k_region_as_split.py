"""
지역(GPS)을 split으로 사용하는 weather 데이터셋 생성.

  train <- New York            (region 0)
  val   <- SF Bay Area         (region 1)
  test  <- Israel/Other/NoGPS  (region 2)

각 줄 포맷: "{image_path} {weather_label}"  (참고 코드와 동일)
지역 분류는 이미 만들어 둔 data/bdd100k_region_{train,val,test}.txt 를 재사용한다
(info JSON을 다시 읽지 않으므로 빠르다). weather 라벨만 labels JSON에서 읽는다.
"""
import re
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# ── paths ─────────────────────────────────────────────────────────────────────
BDD100K_BASE = "/home/young/AILabDataset/01_Open_Dataset/26_BDD-100K"
LABEL_BASE   = os.path.join(BDD100K_BASE, "labels/100k")  # per-image 라벨 JSON

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REGION_TXT = {sp: os.path.join(SCRIPT_DIR, "data", f"bdd100k_region_{sp}.txt")
              for sp in ("train", "val", "test")}  # 입력: 기존 지역 분류 결과

SAVE_DIR = os.path.join(SCRIPT_DIR, "data", "region_as_split")
PLOT_DIR = os.path.join(SAVE_DIR, "plots")

# ── label mapping (BDD100K 전체 weather 클래스, foggy 포함) ───────────────────────
# clear/partly cloudy 는 동일 의미라 0으로 통합. undefined 는 실제 날씨가 아니므로 스킵.
WEATHER_MAP = {
    "clear":         0,
    "partly cloudy": 0,
    "overcast":      1,
    "foggy":         2,
    "rainy":         3,
    "snowy":         4,
}
WEATHER_CLASS_NAMES = {0: "Clear", 1: "Overcast", 2: "Foggy", 3: "Rainy", 4: "Snowy"}
WEATHER_COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# region id -> 목표 split
REGION_TO_SPLIT = {0: "train", 1: "val", 2: "test"}
SPLIT_REGION_NAME = {"train": "New York", "val": "SF Bay Area", "test": "Israel/Other/NoGPS"}

WEATHER_RE = re.compile(r'"weather"\s*:\s*"([^"]+)"')


def load_region_assignments():
    """기존 region txt에서 (image_path, official_split, region_id) 목록을 읽는다."""
    items = []
    for off_split, txt in REGION_TXT.items():
        with open(txt) as f:
            for line in f:
                path, reg = line.rsplit(" ", 1)
                items.append((path, off_split, int(reg)))
    print(f"지역 분류 로드: {len(items):,} 이미지")
    return items


def read_weather(label_path):
    """라벨 JSON에서 weather 문자열을 추출. 없으면 None."""
    try:
        with open(label_path) as f:
            m = WEATHER_RE.search(f.read())
    except FileNotFoundError:
        return None
    return m.group(1) if m else None


def build():
    """지역=split, 라벨=weather 로 train/val/test 데이터를 구성한다."""
    items = load_region_assignments()

    paths = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    n_no_label = n_skip_weather = 0

    for img_path, off_split, region in tqdm(items, desc="weather 라벨링"):
        key = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_BASE, off_split, key + ".json")

        weather = read_weather(label_path)
        if weather is None:
            n_no_label += 1
            continue
        if weather not in WEATHER_MAP:          # undefined 등 매핑 외 값 스킵
            n_skip_weather += 1
            continue

        tgt = REGION_TO_SPLIT[region]
        paths[tgt].append(img_path)
        labels[tgt].append(WEATHER_MAP[weather])

    print(f"\n  라벨 파일 없음: {n_no_label}, weather 스킵(undefined/foggy): {n_skip_weather}")
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
    region = SPLIT_REGION_NAME[split_name]
    print(f"\n[{split_name} = {region}] Weather distribution (total: {total:,})")
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
        region = SPLIT_REGION_NAME[sp]
        plot_distribution(
            labels[sp],
            title=f"BDD100K Weather — {sp} ({region})",
            save_path=os.path.join(PLOT_DIR, f"weather_{sp}.png"),
            class_names=WEATHER_CLASS_NAMES,
            colors=WEATHER_COLORS,
        )

    print("\nDone.")
