"""
Auto-annotation pipeline: MediaPipe Hands → YOLO labels → DETR-ready dataset.

Run from src/utils/ with:
    uv run auto_annotate.py

Flow:
    1. Scan  data/train/images/  for all collected images
    2. Group images by class (parsed from filename: <class>-<uuid>.jpg)
    3. Run MediaPipe Hands on every image → extract bounding box(es)
    4. Write YOLO-format .txt labels to  data/train/labels/
    5. Stratified 80/20 split → move 20% image + label to  data/test/
    6. Update src/config.json with the discovered class list
"""

import cv2
import json
import os
import random
import shutil
from pathlib import Path

import mediapipe as mp
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ── Paths (always relative to this script, regardless of CWD) ─────────────────
SCRIPT_DIR       = Path(__file__).parent            # src/utils/
DATA_DIR         = SCRIPT_DIR / "data"              # src/utils/data/
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
TRAIN_LABELS_DIR = DATA_DIR / "train" / "labels"
TEST_IMAGES_DIR  = DATA_DIR / "test"  / "images"
TEST_LABELS_DIR  = DATA_DIR / "test"  / "labels"
CONFIG_PATH      = SCRIPT_DIR.parent / "config.json"   # src/config.json

# ── Settings ──────────────────────────────────────────────────────────────────
BBOX_PADDING = 0.05    # fractional padding added around hand landmarks
TRAIN_RATIO  = 0.80    # 80 % train / 20 % test
MIN_CONF     = 0.3     # MediaPipe minimum detection confidence
RANDOM_SEED  = 42

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_class(filename: str) -> str:
    """
    Extract class label from  <class_name>-<uuid>.jpg.
    UUID v1 is always 36 chars, so strip the trailing '-<uuid>' (37 chars).
    """
    return Path(filename).stem[:-37]


def landmarks_to_yolo(hand_landmarks) -> tuple[float, float, float, float]:
    """
    Convert 21 normalised MediaPipe landmarks → YOLO (cx, cy, w, h) in [0,1].
    Adds BBOX_PADDING around the tight bounding box.
    """
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x_min = max(0.0, min(xs) - BBOX_PADDING)
    x_max = min(1.0, max(xs) + BBOX_PADDING)
    y_min = max(0.0, min(ys) - BBOX_PADDING)
    y_max = min(1.0, max(ys) + BBOX_PADDING)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return cx, cy, w, h


def scan_images(images_dir: Path) -> dict[str, list[Path]]:
    """Group all .jpg/.png images in images_dir by class name."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            cls = parse_class(f.name)
            groups.setdefault(cls, []).append(f)
        except Exception:
            console.print(f"[yellow]⚠ Skipping unrecognised filename: {f.name}[/yellow]")
    return groups


def annotate_image(
    image_path: Path,
    class_id: int,
    label_path: Path,
    hands,
) -> int:
    """
    Run MediaPipe on one image, write YOLO label file.
    Both detected hands are written as separate annotation lines.
    Returns number of hands annotated (0 = no hand found, file not written).
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return 0

    lines = []
    for hand_lms in results.multi_hand_landmarks:
        cx, cy, w, h = landmarks_to_yolo(hand_lms)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold cyan]🤟 SignDETR — MediaPipe Auto-Annotator[/bold cyan]\n"
        "[dim]Hands detection → YOLO labels → 80/20 split → config update[/dim]",
        border_style="blue",
    ))

    # Ensure directories exist
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_IMAGES_DIR.exists():
        console.print(f"[red]❌ Images directory not found: {TRAIN_IMAGES_DIR}[/red]")
        return

    # ── Step 1: scan images ────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/4[/bold] Scanning images...")
    class_groups = scan_images(TRAIN_IMAGES_DIR)

    if not class_groups:
        console.print("[red]❌ No images found. Run collect_images.py first.[/red]")
        return

    sorted_classes = sorted(class_groups.keys())
    class_to_id    = {cls: idx for idx, cls in enumerate(sorted_classes)}
    total_images   = sum(len(v) for v in class_groups.values())

    console.print(f"  Found [cyan]{len(sorted_classes)}[/cyan] classes, "
                  f"[cyan]{total_images}[/cyan] images total.")

    # ── Step 2: annotate ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/4[/bold] Running MediaPipe annotation...")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=MIN_CONF,
    )

    stats: dict[str, dict] = {}   # {class_name: {total, annotated, skipped}}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[cyan]Annotating", total=total_images)

        for cls in sorted_classes:
            images = class_groups[cls]
            class_id = class_to_id[cls]
            annotated, skipped = 0, 0

            for img_path in images:
                label_path = TRAIN_LABELS_DIR / (img_path.stem + ".txt")
                n = annotate_image(img_path, class_id, label_path, hands_detector)
                if n > 0:
                    annotated += 1
                else:
                    skipped += 1
                progress.advance(overall)

            stats[cls] = {"total": len(images), "annotated": annotated, "skipped": skipped}

    hands_detector.close()

    total_annotated = sum(s["annotated"] for s in stats.values())
    total_skipped   = sum(s["skipped"]   for s in stats.values())
    console.print(f"  ✅ Annotated: [green]{total_annotated}[/green]  "
                  f"⚠ Skipped (no hand): [yellow]{total_skipped}[/yellow]")

    # ── Step 3: 80/20 stratified split ────────────────────────────────────────
    console.print("\n[bold]Step 3/4[/bold] Performing 80/20 train/test split...")

    random.seed(RANDOM_SEED)
    train_count, test_count = 0, 0

    for cls, images in class_groups.items():
        # Only split images that were successfully annotated
        annotated_images = [p for p in images
                            if (TRAIN_LABELS_DIR / (p.stem + ".txt")).exists()]

        random.shuffle(annotated_images)
        split_idx = max(1, int(len(annotated_images) * TRAIN_RATIO))
        test_images = annotated_images[split_idx:]

        for img_path in test_images:
            label_path = TRAIN_LABELS_DIR / (img_path.stem + ".txt")
            # Move image
            shutil.move(str(img_path),   str(TEST_IMAGES_DIR / img_path.name))
            # Move label
            shutil.move(str(label_path), str(TEST_LABELS_DIR / label_path.name))

        train_count += split_idx
        test_count  += len(test_images)

    console.print(f"  Train: [green]{train_count}[/green]  Test: [cyan]{test_count}[/cyan]")

    # ── Step 4: update config.json ────────────────────────────────────────────
    console.print("\n[bold]Step 4/4[/bold] Updating config.json...")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Generate distinct colours (evenly spaced HSV → RGB)
    import colorsys
    n = len(sorted_classes)
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])

    config["classes"] = sorted_classes
    config["colors"]  = colors

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"  Config updated with [cyan]{n}[/cyan] classes.")

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    table = Table(title="📊 Per-Class Annotation Summary", header_style="bold magenta")
    table.add_column("Class",      style="cyan")
    table.add_column("ID",         style="dim")
    table.add_column("Total",      justify="right")
    table.add_column("Annotated",  justify="right", style="green")
    table.add_column("Skipped",    justify="right", style="yellow")
    table.add_column("Train",      justify="right")
    table.add_column("Test",       justify="right")

    for cls in sorted_classes:
        s = stats[cls]
        ann  = s["annotated"]
        split = max(1, int(ann * TRAIN_RATIO))
        tr   = split
        te   = ann - split
        table.add_row(
            cls, str(class_to_id[cls]),
            str(s["total"]), str(ann), str(s["skipped"]),
            str(tr), str(te),
        )

    console.print(table)
    console.print(Panel.fit(
        f"[bold green]🎉 Done![/bold green]  "
        f"[cyan]{total_annotated}[/cyan] images annotated  •  "
        f"[green]{train_count}[/green] train  •  "
        f"[cyan]{test_count}[/cyan] test\n"
        "[dim]Next → run: uv run train.py[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
