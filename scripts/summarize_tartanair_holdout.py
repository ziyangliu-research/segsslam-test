#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rigid_align_se3(est_xyz: np.ndarray, gt_xyz: np.ndarray):
    """Rigid SE(3) alignment only. No scale fitting."""
    if len(est_xyz) < 3:
        return None, None, None

    mu_est = est_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)
    x = est_xyz - mu_est
    y = gt_xyz - mu_gt

    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = mu_gt - r @ mu_est
    aligned = (r @ est_xyz.T).T + t
    return aligned, r, t


def load_eval_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_ate(eval_csv: Path, gt_file: Path):
    rows = load_eval_rows(eval_csv)
    gt = np.loadtxt(gt_file, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt[None, :]
    if gt.shape[1] < 7:
        raise ValueError(f"GT must contain tx ty tz qx qy qz qw: {gt_file}")

    est_xyz = []
    gt_xyz = []
    used_frames = []
    for row in rows:
        if row["tracked"] != "1" or row["largest_map"] != "1":
            continue
        frame = int(row["frame"])
        if frame >= len(gt):
            continue
        try:
            xyz = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
        except ValueError:
            continue
        if not np.isfinite(xyz).all():
            continue
        est_xyz.append(xyz)
        gt_xyz.append(gt[frame, :3])
        used_frames.append(frame)

    if len(est_xyz) < 3:
        return float("nan"), len(est_xyz)

    est_xyz = np.asarray(est_xyz)
    gt_xyz = np.asarray(gt_xyz)
    aligned, _, _ = rigid_align_se3(est_xyz, gt_xyz)
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return rmse, len(est_xyz)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def fmt_ratio(x):
    return f"{100.0 * float(x):.2f}%"


def collect_mode(seq_dir: Path, mode: str, gt_file: Path, fps_stats: dict):
    d = seq_dir / mode
    stats = load_json(d / "run_stats.json")
    render = load_json(d / "render_summary.json")
    ate, ate_frames = compute_ate(d / "per_frame_eval.csv", gt_file)

    return {
        "Sequence": stats["sequence"],
        "MaxMap": fmt_ratio(stats["maxmap_ratio"]),
        "Train PSNR/SSIM": f"{render['train_psnr']:.4f}/{render['train_ssim']:.6f}",
        "Test PSNR/SSIM": f"{render['test_psnr']:.4f}/{render['test_ssim']:.6f}",
        "ATE(m)": f"{ate:.6f}" if np.isfinite(ate) else "nan",
        # FPS is deliberately taken from the separate compute-limited light-mode pass.
        "FPS": f"{float(fps_stats['pipeline_fps']):.4f}",
        # SEGS-SLAM uses k structured Gaussians per anchor; run_stats stores anchors*k.
        "Gaussians": str(int(stats["gaussians"])),
        "Anchors": str(int(stats["anchors"])),
        "ATEFrames": str(int(ate_frames)),
        "Iterations": str(int(stats["final_iteration"])),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Sequence",
        "MaxMap",
        "Train PSNR/SSIM",
        "Test PSNR/SSIM",
        "ATE(m)",
        "FPS",
        "Gaussians",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def print_table(title, rows):
    fields = ["Sequence", "MaxMap", "Train PSNR/SSIM", "Test PSNR/SSIM", "ATE(m)", "FPS", "Gaussians"]
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in fields}
    print("\n" + title)
    print(" | ".join(k.ljust(widths[k]) for k in fields))
    print("-+-".join("-" * widths[k] for k in fields))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(widths[k]) for k in fields))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", default=["SH000", "SH001", "SH002", "SH003"])
    args = p.parse_args()

    online_rows = []
    full_rows = []
    details = {}

    for seq in args.sequences:
        seq_dir = args.result_root / seq
        gt_file = args.data_root / "ground_truth" / "stereo_gt" / f"{seq}.txt"
        if not gt_file.is_file():
            raise FileNotFoundError(gt_file)

        fps_stats = load_json(seq_dir / "fps" / "run_stats.json")
        online = collect_mode(seq_dir, "online", gt_file, fps_stats)
        full = collect_mode(seq_dir, "full30k", gt_file, fps_stats)
        online_rows.append(online)
        full_rows.append(full)
        details[seq] = {"online": online, "full30k": full, "fps_stats": fps_stats}

    write_csv(args.result_root / "summary_online.csv", online_rows)
    write_csv(args.result_root / "summary_full30k.csv", full_rows)
    with (args.result_root / "summary_details.json").open("w") as f:
        json.dump(details, f, indent=2)

    print_table("ONLINE / light-mode map", online_rows)
    print_table("FULL / official 30K map", full_rows)
    print("\nFPS column: compute-limited light-mode pipeline FPS (frames / time until Gaussian mapper exits).")
    print("ATE: rigid SE(3) alignment on successfully tracked frames in the largest ORB-SLAM3 map; no scale fitting.")
    print("Gaussians: anchors x n_offsets (SEGS-SLAM uses k structured Gaussians per anchor).")


if __name__ == "__main__":
    main()
