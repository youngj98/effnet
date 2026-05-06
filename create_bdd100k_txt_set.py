import re
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# ── paths ─────────────────────────────────────────────────────────────────────

BDD100K_BASE = "/home/young/AILabDataset/01_Open_Dataset/09_BDD-100K/bdd100k"
IMAGE_BASE   = os.path.join(BDD100K_BASE, "images/100k")
LABEL_DIR    = os.path.join(BDD100K_BASE, "labels")

TRAIN_JSON = os.path.join(LABEL_DIR, "bdd100k_labels_images_train.json")
VAL_JSON   = os.path.join(LABEL_DIR, "bdd100k_labels_images_val.json")

TEST_SIZE = 5000  # samples to split from train as test set

# ── label mappings ────────────────────────────────────────────────────────────

'''
Weather classes
  0 : clear / partly cloudy
  1 : overcast
  2 : foggy
  3 : rainy
  4 : snowy
  (skip: undefined)
'''
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

'''
Time-of-day classes
  0 : daytime
  1 : night
  (skip: dawn/dusk, undefined)
'''
TIME_MAP = {
    "daytime":   0,
    "night":     1,
}
TIME_CLASS_NAMES = {0: "Daytime", 1: "Night"}
TIME_COLORS      = ["#FFD700", "#FF8C00"]


# ── data loading ──────────────────────────────────────────────────────────────

def parse_name_attributes(json_path):
    """JSON을 regex로 스트리밍 파싱해 (name, weather, timeofday) 트리플 추출."""
    pattern = re.compile(
        r'"name"\s*:\s*"([^"]+)".*?"weather"\s*:\s*"([^"]+)".*?"timeofday"\s*:\s*"([^"]+)"',
        re.DOTALL,
    )
    with open(json_path, "r") as f:
        content = f.read()
    return pattern.findall(content)  # [(name, weather, timeofday), ...]


def build_dataset(json_path, split_name, label_map, label_key):
    """
    label_key : 'weather' | 'time'
    label_map : WEATHER_MAP | TIME_MAP
    해당 attribute가 유효한 항목만 필터링해 반환한다.
    """
    print(f"\n[{split_name}/{label_key}] Parsing {json_path} ...")
    triples = parse_name_attributes(json_path)
    print(f"[{split_name}/{label_key}] Total entries found: {len(triples)}")

    image_paths, labels = [], []
    n_skip_label = n_skip_img = 0

    for name, weather, timeofday in tqdm(triples, desc=f"[{split_name}/{label_key}] filtering"):
        attr_val = weather if label_key == "weather" else timeofday

        if attr_val not in label_map:
            n_skip_label += 1
            continue

        img_path = os.path.join(IMAGE_BASE, split_name, name)
        if not os.path.isfile(img_path):
            n_skip_img += 1
            continue

        image_paths.append(img_path)
        labels.append(label_map[attr_val])

    print(f"  skipped — undefined: {n_skip_label}, missing image: {n_skip_img}")
    print(f"  kept: {len(image_paths):,}")

    return np.array(image_paths), np.array(labels, dtype=np.int32)


# ── test split ────────────────────────────────────────────────────────────────

def split_test_proportional(paths, labels, class_names, test_size=TEST_SIZE):
    """클래스 비율을 유지하면서 test_size개를 분리해 (train, test) 쌍을 반환한다."""
    n_classes    = max(class_names.keys()) + 1
    class_counts = np.array([int(np.sum(labels == i)) for i in range(n_classes)])
    total        = len(labels)

    raw_alloc  = test_size * class_counts / total
    test_alloc = np.floor(raw_alloc).astype(int)
    remainder  = test_size - test_alloc.sum()
    fractions  = raw_alloc - test_alloc
    test_alloc[np.argsort(-fractions)[:remainder]] += 1

    print(f"\n[test split] target {test_size:,}, allocated {test_alloc.sum():,}")
    for i, n in enumerate(test_alloc):
        print(f"  class {i} ({class_names[i]}): {n} / {class_counts[i]}")

    train_idx, test_idx = [], []
    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        n_test = test_alloc[cls]
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return (paths[train_idx], labels[train_idx],
            paths[test_idx],  labels[test_idx])


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_txt(image_paths, labels, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        for img, lbl in zip(image_paths, labels):
            f.write(f"{img} {lbl}\n")
    print(f"  Saved {len(image_paths):,} entries -> {dst}")


# ── plotting ──────────────────────────────────────────────────────────────────

def print_distribution(labels, split_name, class_names):
    total = len(labels)
    print(f"\n[{split_name}] Class distribution (total: {total:,})")
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

    ax.set_xlabel("Class", fontsize=12)
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


# ── pipeline helper ───────────────────────────────────────────────────────────

def run_pipeline(label_key, label_map, class_names, colors, save_dir, plot_dir):
    """weather / time 각각에 대해 독립적으로 파싱 → 분리 → 저장 → 플롯을 실행한다."""
    # train: parse → shuffle → split test
    all_paths, all_labels = build_dataset(TRAIN_JSON, "train", label_map, label_key)

    idx = np.arange(len(all_paths))
    np.random.shuffle(idx)
    all_paths, all_labels = all_paths[idx], all_labels[idx]

    train_paths, train_labels, test_paths, test_labels = \
        split_test_proportional(all_paths, all_labels, class_names, TEST_SIZE)

    # val: parse → shuffle
    val_paths, val_labels = build_dataset(VAL_JSON, "val", label_map, label_key)

    idx = np.arange(len(val_paths))
    np.random.shuffle(idx)
    val_paths, val_labels = val_paths[idx], val_labels[idx]

    # print
    for split, lbl in [("train", train_labels), ("test", test_labels), ("val", val_labels)]:
        print_distribution(lbl, f"{split}/{label_key}", class_names)

    # write txt
    print()
    splits = [("train", train_paths, train_labels),
               ("test",  test_paths,  test_labels),
               ("valid", val_paths,   val_labels)]

    for split, paths, lbl in splits:
        write_txt(paths, lbl,
                  os.path.join(save_dir, f"bdd100k_{label_key}_{split}.txt"))

    # plot
    print()
    for split, _, lbl in splits:
        plot_distribution(
            lbl,
            title=f"BDD100K {label_key.capitalize()} Distribution — {split}",
            save_path=os.path.join(plot_dir, f"bdd100k_{label_key}_{split}.png"),
            class_names=class_names,
            colors=colors,
        )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR = "./data"
    PLOT_DIR = "./data/plots"

    run_pipeline("weather", WEATHER_MAP, WEATHER_CLASS_NAMES, WEATHER_COLORS, SAVE_DIR, PLOT_DIR)
    run_pipeline("time",    TIME_MAP,    TIME_CLASS_NAMES,    TIME_COLORS,    SAVE_DIR, PLOT_DIR)

    print("\nDone.")
