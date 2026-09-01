#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rigid_align_se3(est_xyz, gt_xyz):
    mu_e = est_xyz.mean(axis=0)
    mu_g = gt_xyz.mean(axis=0)
    x = est_xyz - mu_e
    y = gt_xyz - mu_g
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    t = mu_g - r @ mu_e
    return (r @ est_xyz.T).T + t


def compute_ate(eval_csv: Path, gt_file: Path):
    rows = load_rows(eval_csv)
    gt = np.loadtxt(gt_file, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt[None, :]
    est, target = [], []
    for row in rows:
        if row["tracked"] != "1" or row["largest_map"] != "1":
            continue
        frame = int(row["frame"])
        if frame >= len(gt):
            continue
        xyz = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
        if not np.isfinite(xyz).all():
            continue
        est.append(xyz)
        target.append(gt[frame, :3])
    if len(est) < 3:
        return float("nan"), len(est)
    est = np.asarray(est)
    target = np.asarray(target)
    aligned = rigid_align_se3(est, target)
    err = np.linalg.norm(aligned - target, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), len(est)


def collect(seq, mode_dir: Path, gt_file: Path, fps_stats):
    stats = load_json(mode_dir / "run_stats.json")
    render = load_json(mode_dir / "render_summary.json")
    ate, ate_frames = compute_ate(mode_dir / "per_frame_eval.csv", gt_file)
    return {
        "Sequence": seq,
        "MaxMap": f"{100.0 * float(stats['maxmap_ratio']):.2f}%",
        "Train PSNR/SSIM": f"{float(render['train_psnr']):.4f}/{float(render['train_ssim']):.6f}",
        "Test PSNR/SSIM": f"{float(render['test_psnr']):.4f}/{float(render['test_ssim']):.6f}",
        "ATE(m)": f"{ate:.6f}" if np.isfinite(ate) else "nan",
        "FPS": f"{float(fps_stats['pipeline_fps']):.4f}",
        "Gaussians": str(int(stats["gaussians"])),
        "Iterations": int(stats["final_iteration"]),
        "ATEFrames": ate_frames,
    }


def write_csv(path, rows):
    fields = ["Sequence", "MaxMap", "Train PSNR/SSIM", "Test PSNR/SSIM", "ATE(m)", "FPS", "Gaussians"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def print_table(title, rows):
    fields = ["Sequence", "MaxMap", "Train PSNR/SSIM", "Test PSNR/SSIM", "ATE(m)", "FPS", "Gaussians"]
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in fields}
    print("\n" + title)
    print(" | ".join(k.ljust(widths[k]) for k in fields))
    print("-+-".join("-" * widths[k] for k in fields))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(widths[k]) for k in fields))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", default=["SH000", "SH001", "SH002", "SH003"])
    args = ap.parse_args()

    online_rows, full_rows = [], []
    details = {}
    for seq in args.sequences:
        seq_dir = args.result_root / seq
        dual = seq_dir / "dual"
        fps_path = seq_dir / "fps" / "run_stats.json"
        if not fps_path.is_file():
            raise FileNotFoundError(f"Missing FPS pass: {fps_path}")
        fps_stats = load_json(fps_path)
        gt = args.data_root / "ground_truth" / "stereo_gt" / f"{seq}.txt"
        online = collect(seq, dual / "online", gt, fps_stats)
        full = collect(seq, dual / "full30k", gt, fps_stats)

        # A dual run must share exactly the same tracking result.
        if online["MaxMap"] != full["MaxMap"] or online["ATE(m)"] != full["ATE(m)"]:
            raise RuntimeError(f"{seq}: dual snapshot tracking mismatch; this should be impossible")

        online_rows.append(online)
        full_rows.append(full)
        details[seq] = {
            "online": online,
            "full30k": full,
            "fps_stats": fps_stats,
            "common": load_json(dual / "common_stats.json"),
        }

    write_csv(args.result_root / "summary_dual_online.csv", online_rows)
    write_csv(args.result_root / "summary_dual_full30k.csv", full_rows)
    with (args.result_root / "summary_dual_details.json").open("w") as f:
        json.dump(details, f, indent=2)

    print_table("ONLINE SNAPSHOT / same run", online_rows)
    print_table("FULL 30K SNAPSHOT / same run", full_rows)
    print("\nMaxMap and ATE are intentionally identical between the two tables: same tracking run, same poses.")
    print("FPS reuses the existing compute-limited light-mode speed pass to avoid another long rerun.")


if __name__ == "__main__":
    main()
