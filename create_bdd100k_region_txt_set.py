import re
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# ── paths ─────────────────────────────────────────────────────────────────────
# info(센서/GPS) JSON과 이미지가 같은 해시 파일명을 공유한다.
#   images : <IMAGE_BASE>/<split>/<key>.jpg
#   info   : <INFO_BASE>/<split>/<key>.json

BDD100K_BASE = "/home/young/AILabDataset/01_Open_Dataset/26_BDD-100K"
IMAGE_BASE   = os.path.join(BDD100K_BASE, "images/100k")
INFO_BASE    = os.path.join(BDD100K_BASE, "100k")  # info 압축 해제 위치

SPLITS = ["train", "val", "test"]  # 지역 라벨(GPS)은 세 split 모두 존재

# ── region label mapping ────────────────────────────────────────────────────────
'''
Region classes (GPS 좌표 기반)
  0 : New York            (NYC 광역권)
  1 : SF Bay Area         (San Francisco / Berkeley / Bay Area)
  2 : Israel / Other / NoGPS  (그 외 전부 + GPS 없음)
'''
REGION_CLASS_NAMES = {0: "New York", 1: "SF Bay Area", 2: "Israel/Other/NoGPS"}
REGION_COLORS      = ["#4C72B0", "#55A868", "#C44E52"]

# 광역권 바운딩박스 (lat_min, lat_max, lon_min, lon_max)
NYC_BOX = (40.0, 42.5, -75.5, -73.0)
SF_BOX  = (36.8, 38.8, -123.5, -121.2)


def classify_region(lat, lon):
    """대표 GPS 좌표를 3개 지역 클래스로 분류한다. 좌표 없으면 2."""
    if lat is None or lon is None:
        return 2
    la0, la1, lo0, lo1 = NYC_BOX
    if la0 <= lat <= la1 and lo0 <= lon <= lo1:
        return 0
    la0, la1, lo0, lo1 = SF_BOX
    if la0 <= lat <= la1 and lo0 <= lon <= lo1:
        return 1
    return 2


# ── data loading ──────────────────────────────────────────────────────────────
# 40초 클립은 한 도시권을 벗어나지 않으므로 첫 유효 좌표 하나면 충분하다.
# latitude/longitude는 거대한 가속도/자이로 배열 뒤(locations/gps)에 나오므로
# json.load 대신 regex로 첫 좌표만 뽑는다.
LAT_RE = re.compile(r'"latitude"\s*:\s*(-?\d+(?:\.\d+)?)')
LON_RE = re.compile(r'"longitude"\s*:\s*(-?\d+(?:\.\d+)?)')


def rep_coord(info_path):
    """info JSON에서 대표 (lat, lon)을 추출. 없거나 (0,0)이면 (None, None)."""
    with open(info_path, "r") as f:
        content = f.read()
    mlat = LAT_RE.search(content)
    mlon = LON_RE.search(content)
    if not (mlat and mlon):
        return None, None
    lat, lon = float(mlat.group(1)), float(mlon.group(1))
    if lat == 0 and lon == 0:
        return None, None
    return lat, lon


def build_dataset(split):
    """해당 split의 모든 이미지를 (image_path, region_label)로 변환한다."""
    info_dir = os.path.join(INFO_BASE, split)
    img_dir  = os.path.join(IMAGE_BASE, split)
    files    = sorted(glob.glob(os.path.join(info_dir, "*.json")))
    print(f"\n[{split}] info files found: {len(files):,}")

    image_paths, labels = [], []
    n_missing_img = n_nogps = 0

    for fp in tqdm(files, desc=f"[{split}] classifying"):
        key      = os.path.splitext(os.path.basename(fp))[0]
        img_path = os.path.join(img_dir, key + ".jpg")
        if not os.path.isfile(img_path):
            n_missing_img += 1
            continue

        lat, lon = rep_coord(fp)
        if lat is None:
            n_nogps += 1
        image_paths.append(img_path)
        labels.append(classify_region(lat, lon))

    print(f"  missing image: {n_missing_img}, no-GPS(→class 2): {n_nogps}")
    print(f"  kept: {len(image_paths):,}")
    return np.array(image_paths), np.array(labels, dtype=np.int32)


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
    print(f"\n[{split_name}] Region distribution (total: {total:,})")
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Region", fontsize=12)
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


def plot_grouped(split_labels, title, save_path, class_names, colors):
    """split별(train/val/test)로 클래스 분포를 묶은 그룹 막대그래프."""
    n_classes = max(class_names.keys()) + 1
    splits    = list(split_labels.keys())
    x         = np.arange(n_classes)
    width     = 0.8 / len(splits)

    _, ax = plt.subplots(figsize=(11, 6))
    for i, split in enumerate(splits):
        lbl    = split_labels[split]
        counts = [int(np.sum(lbl == c)) for c in range(n_classes)]
        offset = (i - (len(splits) - 1) / 2) * width
        bars   = ax.bar(x + offset, counts, width, label=split, edgecolor="white", linewidth=0.6)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{c:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in range(n_classes)])
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="split")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR   = os.path.join(SCRIPT_DIR, "data")
    PLOT_DIR   = os.path.join(SAVE_DIR, "plots")

    # 1) 각 split을 GPS로 지역 분류
    split_paths, split_labels = {}, {}
    for split in SPLITS:
        paths, labels = build_dataset(split)
        split_paths[split], split_labels[split] = paths, labels

    # 2) 분포 출력
    for split in SPLITS:
        print_distribution(split_labels[split], split, REGION_CLASS_NAMES)

    # 3) txt 저장 ( {image_path} {region_label} )
    print()
    for split in SPLITS:
        write_txt(split_paths[split], split_labels[split],
                  os.path.join(SAVE_DIR, f"bdd100k_region_{split}.txt"))

    # 4) split별 개별 그래프 + 통합 그룹 그래프
    print()
    for split in SPLITS:
        plot_distribution(
            split_labels[split],
            title=f"BDD100K Region Distribution — {split}",
            save_path=os.path.join(PLOT_DIR, f"bdd100k_region_{split}.png"),
            class_names=REGION_CLASS_NAMES,
            colors=REGION_COLORS,
        )
    plot_grouped(
        split_labels,
        title="BDD100K Region Distribution — train/val/test",
        save_path=os.path.join(PLOT_DIR, "bdd100k_region_all.png"),
        class_names=REGION_CLASS_NAMES,
        colors=REGION_COLORS,
    )

    print("\nDone.")
