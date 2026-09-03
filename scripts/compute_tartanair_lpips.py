#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import lpips
except ImportError as exc:
    raise SystemExit(
        "ERROR: Python package 'lpips' is missing. Install it with:\n"
        "  python -m pip install lpips"
    ) from exc


def image_tensor(path: Path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def make_model(device):
    model = lpips.LPIPS(net="alex").to(device).eval()
    return model


def preflight():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(device)
    x = torch.zeros(1, 3, 64, 64, device=device)
    with torch.no_grad():
        y = model(x, x)
    print(f"LPIPS preflight OK: device={device}, value={float(y.item()):.6f}")


def compute(snapshot_dir: Path, data_root: Path, sequence: str, batch_size: int):
    rows_path = snapshot_dir / "per_frame_eval.csv"
    renders = snapshot_dir / "renders"
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)
    if not renders.is_dir():
        raise FileNotFoundError(renders)

    with rows_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    items = []
    for row in rows:
        if row["largest_map"] != "1":
            continue
        frame = int(row["frame"])
        render = renders / f"{frame:06d}.png"
        gt = data_root / "stereo" / sequence / "image_left" / f"{frame:06d}_left.png"
        if render.is_file() and gt.is_file():
            items.append((frame, row["split"], render, gt))

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

    train = [values[f] for f, split, _, _ in items if split == "train"]
    test = [values[f] for f, split, _, _ in items if split == "test"]
    summary = {
        "network": "alex",
        "train_count": len(train),
        "train_lpips": float(np.mean(train)) if train else float("nan"),
        "test_count": len(test),
        "test_lpips": float(np.mean(test)) if test else float("nan"),
    }
    with (snapshot_dir / "lpips_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    per_frame = snapshot_dir / "per_frame_lpips.csv"
    with per_frame.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "split", "lpips"])
        for frame, split, _, _ in items:
            w.writerow([frame, split, f"{values[frame]:.9f}"])

    render_summary_path = snapshot_dir / "render_summary.json"
    if render_summary_path.is_file():
        render_summary = json.loads(render_summary_path.read_text())
        render_summary["train_lpips"] = summary["train_lpips"]
        render_summary["test_lpips"] = summary["test_lpips"]
        render_summary["lpips_network"] = "alex"
        render_summary_path.write_text(json.dumps(render_summary, indent=2) + "\n")

    print(
        f"LPIPS {sequence} {snapshot_dir.name}: "
        f"train={summary['train_lpips']:.6f}, test={summary['test_lpips']:.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--snapshot-dir", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--sequence")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.preflight:
        preflight()
        return
    if not args.snapshot_dir or not args.data_root or not args.sequence:
        ap.error("--snapshot-dir, --data-root and --sequence are required unless --preflight is used")
    compute(args.snapshot_dir, args.data_root, args.sequence, args.batch_size)


if __name__ == "__main__":
    main()
