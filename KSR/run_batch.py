#!/usr/bin/env python3
"""
Batch runner for ksr_modified.
Finds all planar_mesh_for_KSR.ply files under AdaptivePS-sparse/**/
and runs ksr_modified on each, writing output to ksr_output/ next to the PLY.
"""

import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent          # PlanarSplatting/
BINARY      = REPO_ROOT / "KSR" / "build" / "ksr_modified"
SEARCH_ROOT = REPO_ROOT / "AdaptivePS-sparse"
PLY_NAME    = "planar_mesh_for_KSR.ply"
OUTPUT_DIR  = "ksr_output"
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if not BINARY.exists():
        sys.exit(f"[ERROR] Binary not found: {BINARY}\n"
                 "  Build it first:  cmake --build KSR/build")

    ply_files = sorted(SEARCH_ROOT.rglob(PLY_NAME))
    if not ply_files:
        sys.exit(f"[ERROR] No '{PLY_NAME}' files found under {SEARCH_ROOT}")

    print(f"Found {len(ply_files)} input file(s):\n")
    for p in ply_files:
        print(f"  {p.relative_to(REPO_ROOT)}")
    print()

    results = []
    for ply in ply_files:
        out_dir = ply.parent / OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        label = str(ply.relative_to(REPO_ROOT))
        print(f"[RUN] {label}")
        print(f"      → {out_dir.relative_to(REPO_ROOT)}/")

        cmd = [str(BINARY), "-i", str(ply), "-o", str(out_dir)]
        result = subprocess.run(cmd, capture_output=False, text=True)

        status = "✓ OK" if result.returncode == 0 else f"✗ FAILED (exit {result.returncode})"
        print(f"      {status}\n")
        results.append((label, result.returncode))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    ok = sum(1 for _, rc in results if rc == 0)
    for label, rc in results:
        icon = "✓" if rc == 0 else "✗"
        print(f"  {icon}  {label}")
    print(f"\n{ok}/{len(results)} succeeded.")


if __name__ == "__main__":
    main()
