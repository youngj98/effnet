"""
scene(장면 유형)을 split으로 사용하는 time(주/야) 데이터셋 생성.

  train <- city street
  val   <- highway
  test  <- 나머지 전부 (residential / parking lot / gas stations / tunnel / undefined 등)

각 줄 포맷: "{image_path} {timeofday_label}"
create_bdd100k_scene_as_split.py 와 split 규칙이 완전히 동일하고, 라벨만
attributes.weather -> attributes.timeofday 로 바뀐 버전이다.

라벨 JSON은 attributes 블록이 파일 끝에 있어 전체를 읽어야 하는데, 데이터셋
마운트가 느려(단일 프로세스 약 17 files/s = 100k 파일에 96분) 프로세스 풀로
병렬 읽기를 한다. 그 외 로직은 weather 버전과 같다.
"""
import re
import os
import argparse
import multiprocessing as mp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# ── paths ─────────────────────────────────────────────────────────────────────
# 라벨은 이 PC에서 읽고, txt에 기록할 이미지 경로는 학습 PC 기준으로 쓴다.
# (기존 data/scene_as_split/*.txt 가 /home/ailab/... 경로를 쓰고 있어 그에 맞춤)
DEFAULT_LABEL_BASE = "/home/young/AILabDataset/01_Open_Dataset/26_BDD-100K/labels/100k"
DEFAULT_IMAGE_BASE = "/home/ailab/09_BDD-100K/bdd100k/images/100k"
OFFICIAL_SPLITS = ("train", "val", "test")  # BDD100K 공식 split (label 폴더 3개 모두 존재)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "data", "scene_as_split_time")

# ── label mapping (BDD100K timeofday) ─────────────────────────────────────────
# dawn/dusk 는 주/야 어느 쪽으로도 단정할 수 없어 제외한다 (2-class 모델과 정합).
# 3-class 로 쓰려면 아래 주석 블록으로 교체.
# TIME_MAP = {
#     "daytime":   0,
#     "dawn/dusk": 1,
#     "night":     2,
# }
# TIME_CLASS_NAMES = {0: "Daytime", 1: "Dawn/Dusk", 2: "Night"}
# TIME_COLORS      = ["#DD8452", "#8172B2", "#4C72B0"]
TIME_MAP = {
    "daytime": 0,
    "night":   1,
}
TIME_CLASS_NAMES = {0: "Daytime", 1: "Night"}
TIME_COLORS      = ["#DD8452", "#4C72B0"]

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

TIMEOFDAY_RE = re.compile(r'"timeofday"\s*:\s*"([^"]+)"')
SCENE_RE     = re.compile(r'"scene"\s*:\s*"([^"]+)"')


def _read_one(task):
    """(off_split, label_path, key) -> (off_split, key, scene, timeofday)"""
    off_split, label_path, key = task
    try:
        with open(label_path) as f:
            text = f.read()
    except OSError:
        return off_split, key, "undefined", None
    m_scene = SCENE_RE.search(text)
    m_tod = TIMEOFDAY_RE.search(text)
    return (
        off_split,
        key,
        m_scene.group(1) if m_scene else "undefined",
        m_tod.group(1) if m_tod else None,
    )


def iter_labels(label_base, workers):
    """BDD100K 라벨 JSON 전체를 병렬로 읽어 (off_split, key, scene, timeofday) yield."""
    tasks = []
    for off_split in OFFICIAL_SPLITS:
        split_dir = os.path.join(label_base, off_split)
        for fname in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            key = os.path.splitext(fname)[0]
            tasks.append((off_split, os.path.join(split_dir, fname), key))

    print(f"  라벨 파일 {len(tasks):,}개, 프로세스 {workers}개로 병렬 읽기")
    with mp.Pool(workers) as pool:
        for rec in tqdm(pool.imap_unordered(_read_one, tasks, chunksize=64),
                        total=len(tasks), desc="scene/timeofday 라벨링"):
            yield rec


def build(label_base, image_base, workers):
    """scene=split, 라벨=timeofday 로 train/val/test 데이터를 구성한다."""
    paths = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    n_no_label = n_skip_time = 0
    skipped = {}

    for off_split, key, scene, tod in iter_labels(label_base, workers):
        if tod is None:
            n_no_label += 1
            continue
        if tod not in TIME_MAP:             # dawn/dusk, undefined 등 매핑 외 값 스킵
            n_skip_time += 1
            skipped[tod] = skipped.get(tod, 0) + 1
            continue

        tgt = SCENE_TO_SPLIT.get(scene, DEFAULT_SPLIT)
        paths[tgt].append(os.path.join(image_base, off_split, key + ".jpg"))
        labels[tgt].append(TIME_MAP[tod])

    detail = ", ".join(f"{k}={v:,}" for k, v in sorted(skipped.items()))
    print(f"\n  라벨 파일 없음: {n_no_label:,}, timeofday 스킵: {n_skip_time:,} ({detail})")

    # 병렬 처리라 순서가 섞이므로 경로 기준으로 정렬해 재현성 확보
    out_p, out_l = {}, {}
    for s in paths:
        order = np.argsort(np.array(paths[s]))
        out_p[s] = np.array(paths[s])[order]
        out_l[s] = np.array(labels[s], dtype=np.int32)[order]
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
    print(f"\n[{split_name} = {scene}] Time distribution (total: {total:,})")
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

    ax.set_xlabel("Time of day", fontsize=12)
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
    ap = argparse.ArgumentParser(description="scene을 split으로 쓰는 time 데이터셋 생성")
    ap.add_argument("--label-base", default=DEFAULT_LABEL_BASE,
                    help="BDD100K per-image 라벨 JSON 루트 (읽기, 이 PC 경로)")
    ap.add_argument("--image-base", default=DEFAULT_IMAGE_BASE,
                    help="txt에 기록할 이미지 루트 (학습 PC 경로)")
    ap.add_argument("--out", default=DEFAULT_SAVE_DIR, help="출력 디렉토리")
    ap.add_argument("--workers", type=int, default=16, help="병렬 읽기 프로세스 수")
    args = ap.parse_args()

    save_dir = args.out
    plot_dir = os.path.join(save_dir, "plots")

    paths, labels = build(args.label_base, args.image_base, args.workers)

    for sp in ("train", "val", "test"):
        print_distribution(labels[sp], sp, TIME_CLASS_NAMES)

    print()
    for sp in ("train", "val", "test"):
        write_txt(paths[sp], labels[sp], os.path.join(save_dir, f"{sp}.txt"))

    print()
    for sp in ("train", "val", "test"):
        scene = SPLIT_SCENE_NAME[sp]
        plot_distribution(
            labels[sp],
            title=f"BDD100K Time of Day — {sp} ({scene})",
            save_path=os.path.join(plot_dir, f"time_{sp}.png"),
            class_names=TIME_CLASS_NAMES,
            colors=TIME_COLORS,
        )

    print("\nDone.")
