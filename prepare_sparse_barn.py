#!/usr/bin/env python3
"""
Prepare sparse-view Barn subsets for TNT evaluation.

For each count in COUNTS:
  - Interval-samples images by int(filename)
  - Creates Barn_sparse{N}/images/
  - Creates Barn_sparse{N}/Barn_COLMAP_SfM.log   (sampled + re-indexed 0,1,...)
  - Creates Barn_sparse{N}/sparse/0/images.txt    (sampled)
  - Copies  Barn_sparse{N}/sparse/0/cameras.txt   (shared camera, unchanged)
  - Copies  eval GT files: Barn.ply / Barn.json / Barn_trans.txt
"""

import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SRC_DIR  = Path("/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet/Barn")
DST_BASE = Path("/Users/mchu/Documents/TUD/Thesis/TNT_GOF/TrainingSet")
COUNTS   = [25, 50, 100, 200, 400]

# Evaluation GT files that need to live alongside the reconstruction
EVAL_COPY = ["Barn.ply", "Barn.json", "Barn_trans.txt"]
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_sfm_log(path: Path):
    """
    Parse a TNT/COLMAP SfM .log file.
    Format: every 5 lines = 1 camera:
        i j 0
        [4×4 matrix, row by row]
    Returns list of (frame_idx: int, matrix_lines: list[str]).
    """
    entries = []
    lines = path.read_text().splitlines()
    assert len(lines) % 5 == 0, f"Expected multiple of 5 lines, got {len(lines)}"
    for i in range(0, len(lines), 5):
        header = lines[i]
        mat    = lines[i+1:i+5]
        frame  = int(header.split()[0])
        entries.append((frame, mat))
    return entries


def write_sfm_log(entries, path: Path):
    """
    Write SfM log. Re-indexes frames as 0, 1, 2, ... regardless of original indices.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for new_idx, (_orig_frame, mat) in enumerate(entries):
            f.write(f"{new_idx} {new_idx} 0\n")
            for row in mat:
                f.write(row + "\n")


def read_images_txt(path: Path):
    """
    Parse COLMAP images.txt.  Returns list of (pose_line: str, pts_line: str).
    Also returns a dict: filename → (pose_line, pts_line).
    """
    entries   = []
    name_map  = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or not lines[i].strip():
            i += 1
            continue
        pose_line = lines[i]
        pts_line  = lines[i+1] if i+1 < len(lines) else ""
        entries.append((pose_line, pts_line))
        # last token on pose line is the image filename
        name = pose_line.strip().split()[-1]
        name_map[name] = (pose_line, pts_line)
        i += 2
    return entries, name_map


def write_images_txt(entries, path: Path):
    """
    Write COLMAP images.txt.
    Keeps original IMAGE_IDs (COLMAP internal IDs — no need to renumber).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(entries)}\n")
        for pose_line, pts_line in entries:
            f.write(pose_line + "\n")
            f.write(pts_line + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Collect & sort all source images by numeric filename
    images_dir = SRC_DIR / "images"
    all_images = sorted(
        [f for f in images_dir.iterdir() if f.is_file() and not f.name.startswith(".")],
        key=lambda p: int(p.stem)
    )
    total = len(all_images)
    print(f"Source: {total} images in {images_dir}")

    # 2. Read full SfM log (410 entries, 1 per image in sorted filename order)
    sfm_path    = SRC_DIR / "Barn_COLMAP_SfM.log"
    sfm_entries = read_sfm_log(sfm_path)
    assert len(sfm_entries) == total, (
        f"SfM log has {len(sfm_entries)} entries but found {total} images — mismatch!"
    )
    print(f"SfM log:  {len(sfm_entries)} entries ✓")

    # 3. Read full images.txt
    images_txt_path = SRC_DIR / "sparse" / "0" / "images.txt"
    _, name_map = read_images_txt(images_txt_path)
    print(f"images.txt: {len(name_map)} entries")

    # 4. Process each sparse count
    for count in COUNTS:
        step    = total / count
        # Uniform interval sampling — pick 'count' indices spread over [0, total)
        indices = sorted({min(int(round(k * step)), total - 1) for k in range(count)})
        sampled_images = [all_images[i] for i in indices]
        n = len(sampled_images)

        dst = DST_BASE / f"Barn_sparse{count}"
        print(f"\n{'─'*60}")
        print(f"  Barn_sparse{count}:  {n} images (step ≈ {step:.2f})")
        print(f"  → {dst}")

        # 4a. Copy images
        img_dst = dst / "images"
        img_dst.mkdir(parents=True, exist_ok=True)
        for img in sampled_images:
            shutil.copy2(img, img_dst / img.name)
        print(f"  Copied {n} images")

        # 4b. Copy evaluation GT files (Barn.ply / Barn.json / Barn_trans.txt)
        for fname in EVAL_COPY:
            src = SRC_DIR / fname
            if src.exists():
                shutil.copy2(src, dst / fname)
            else:
                print(f"  [WARN] not found: {src}")
        print(f"  Copied eval GT files: {EVAL_COPY}")

        # 4c. Sampled SfM log — re-indexed 0,1,...,N-1
        sampled_sfm = [sfm_entries[i] for i in indices]
        write_sfm_log(sampled_sfm, dst / "Barn_COLMAP_SfM.log")
        print(f"  Wrote Barn_COLMAP_SfM.log ({n} entries, re-indexed 0…{n-1})")

        # 4d. Sparse COLMAP: cameras.txt (unchanged) + sampled images.txt
        sparse_dst = dst / "sparse" / "0"
        sparse_dst.mkdir(parents=True, exist_ok=True)

        shutil.copy2(SRC_DIR / "sparse" / "0" / "cameras.txt", sparse_dst / "cameras.txt")
        print(f"  Copied sparse/0/cameras.txt")

        sampled_entries = []
        missing = []
        for img in sampled_images:
            if img.name in name_map:
                sampled_entries.append(name_map[img.name])
            else:
                missing.append(img.name)

        if missing:
            print(f"  [WARN] {len(missing)} images not in images.txt: {missing[:3]}...")

        write_images_txt(sampled_entries, sparse_dst / "images.txt")
        print(f"  Wrote sparse/0/images.txt ({len(sampled_entries)} entries)")

    print(f"\n{'='*60}")
    print("Done. Created:")
    for count in COUNTS:
        dst = DST_BASE / f"Barn_sparse{count}"
        print(f"  {dst}")


if __name__ == "__main__":
    main()
