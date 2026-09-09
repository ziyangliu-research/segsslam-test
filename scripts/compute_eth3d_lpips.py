#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import lpips
except ImportError as exc:
    raise SystemExit(
        "ERROR: Python package 'lpips' is missing. Install without changing torch:\n"
        "  python -m pip install --no-deps lpips==0.1.4"
    ) from exc


def image_tensor(path: Path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def make_model(device):
    return lpips.LPIPS(net="alex").to(device).eval()


def preflight():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(device)
    x = torch.zeros(1, 3, 64, 64, device=device)
    with torch.no_grad():
        y = model(x, x)
    print(f"LPIPS preflight OK: device={device}, value={float(y.item()):.6f}")


def read_manifest(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["frame"]): r for r in rows}


def compute(snapshot_dir: Path, manifest_path: Path, batch_size: int, save_test: bool):
    rows_path = snapshot_dir / "per_frame_eval.csv"
    renders = snapshot_dir / "renders"
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)
    if not renders.is_dir():
        raise FileNotFoundError(renders)
    manifest = read_manifest(manifest_path)

    with rows_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    items = []
    for row in rows:
        if row["largest_map"] != "1":
            continue
        frame = int(row["frame"])
        if frame not in manifest:
            continue
        render = renders / f"{frame:06d}.png"
        gt = Path(manifest[frame]["left_source"])
        if render.is_file() and gt.is_file():
            items.append((frame, row["split"], render, gt, manifest[frame]["source_timestamp"]))

    if not items:
        raise RuntimeError(f"No LPIPS image pairs found in {snapshot_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(device)
    values = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        a = torch.stack([image_tensor(x[2]) for x in batch]).to(device)
        b = torch.stack([image_tensor(x[3]) for x in batch]).to(device)
        with torch.no_grad():
            scores = model(a, b).view(-1).detach().cpu().numpy()
        for item, score in zip(batch, scores):
            values[item[0]] = float(score)
        print(f"[LPIPS] {min(start + len(batch), len(items))}/{len(items)}", end="\r", flush=True)
    print()

    train = [values[f] for f, split, *_ in items if split == "train"]
    test = [values[f] for f, split, *_ in items if split == "test"]
    summary = {
        "network": "alex",
        "train_count": len(train),
        "train_lpips": float(np.mean(train)) if train else float("nan"),
        "test_count": len(test),
        "test_lpips": float(np.mean(test)) if test else float("nan"),
    }
    (snapshot_dir / "lpips_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (snapshot_dir / "per_frame_lpips.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "split", "source_timestamp", "lpips"])
        for frame, split, _, _, stamp in items:
            w.writerow([frame, split, stamp, f"{values[frame]:.9f}"])

    render_summary_path = snapshot_dir / "render_summary.json"
    if render_summary_path.is_file():
        render_summary = json.loads(render_summary_path.read_text())
        render_summary["train_lpips"] = summary["train_lpips"]
        render_summary["test_lpips"] = summary["test_lpips"]
        render_summary["lpips_network"] = "alex"
        render_summary_path.write_text(json.dumps(render_summary, indent=2) + "\n")

    if save_test:
        test_render_dir = snapshot_dir / "test_renders"
        test_gt_dir = snapshot_dir / "test_gt"
        test_render_dir.mkdir(exist_ok=True)
        test_gt_dir.mkdir(exist_ok=True)
        with (snapshot_dir / "test_render_manifest.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "source_timestamp", "render", "gt", "lpips"])
            for frame, split, render, gt, stamp in items:
                if split != "test":
                    continue
                safe_stamp = stamp.replace("/", "_")
                name = f"{frame:06d}__{safe_stamp}.png"
                dst_r = test_render_dir / name
                dst_g = test_gt_dir / name
                shutil.copy2(render, dst_r)
                shutil.copy2(gt, dst_g)
                w.writerow([frame, stamp, str(dst_r), str(dst_g), f"{values[frame]:.9f}"])

    print(
        f"LPIPS {snapshot_dir}: train={summary['train_lpips']:.6f}, "
        f"test={summary['test_lpips']:.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--snapshot-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-test-renders", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        preflight()
        return
    if not args.snapshot_dir or not args.manifest:
        ap.error("--snapshot-dir and --manifest are required unless --preflight is used")
    compute(args.snapshot_dir, args.manifest, args.batch_size, args.save_test_renders)


if __name__ == "__main__":
    main()
