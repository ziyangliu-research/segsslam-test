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


def collect(seq, mode, mode_dir: Path, gt_file: Path, common):
    stats = load_json(mode_dir / "run_stats.json")
    render = load_json(mode_dir / "render_summary.json")
    if "train_lpips" not in render or "test_lpips" not in render:
        raise RuntimeError(f"LPIPS missing from {mode_dir / 'render_summary.json'}")
    ate, ate_frames = compute_ate(mode_dir / "per_frame_eval.csv", gt_file)

    online_t = float(common["online_total_seconds"])
    offline_t = float(common["offline_optimization_seconds"])
    total_t = online_t if mode == "online" else float(common["total_compute_seconds"])

    return {
        "Sequence": seq,
        "MaxMap": f"{100.0 * float(stats['maxmap_ratio']):.2f}%",
        "Train PSNR/SSIM/LPIPS": (
            f"{float(render['train_psnr']):.4f}/"
            f"{float(render['train_ssim']):.6f}/"
            f"{float(render['train_lpips']):.6f}"
        ),
        "Test PSNR/SSIM/LPIPS": (
            f"{float(render['test_psnr']):.4f}/"
            f"{float(render['test_ssim']):.6f}/"
            f"{float(render['test_lpips']):.6f}"
        ),
        "ATE(m)": f"{ate:.6f}" if np.isfinite(ate) else "nan",
        "FPS": f"{float(common['online_fps']):.4f}" if mode == "online" else "-",
        "Online Time(s)": f"{online_t:.3f}",
        "Offline Opt(s)": "-" if mode == "online" else f"{offline_t:.3f}",
        "Total Time(s)": f"{total_t:.3f}",
        "Gaussians": str(int(stats["gaussians"])),
        "Iterations": int(stats["final_iteration"]),
        "ATEFrames": ate_frames,
    }


def fields():
    return [
        "Sequence",
        "MaxMap",
        "Train PSNR/SSIM/LPIPS",
        "Test PSNR/SSIM/LPIPS",
        "ATE(m)",
        "FPS",
        "Online Time(s)",
        "Offline Opt(s)",
        "Total Time(s)",
        "Gaussians",
    ]


def write_csv(path, rows):
    fs = fields()
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fs)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fs})


def print_table(title, rows):
    fs = fields()
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in fs}
    print("\n" + title)
    print(" | ".join(k.ljust(widths[k]) for k in fs))
    print("-+-".join("-" * widths[k] for k in fs))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(widths[k]) for k in fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument(
        "--sequences", nargs="+",
        default=["SE000", "SE001", "SE002", "SE003", "SH000", "SH001", "SH002", "SH003"]
    )
    args = ap.parse_args()

    online_rows, full_rows, details = [], [], {}
    for seq in args.sequences:
        root = args.result_root / seq / "benchmark"
        common = load_json(root / "common_stats.json")
        gt = args.data_root / "ground_truth" / "stereo_gt" / f"{seq}.txt"
        online = collect(seq, "online", root / "online", gt, common)
        full = collect(seq, "full30k", root / "full30k", gt, common)

        if online["MaxMap"] != full["MaxMap"] or online["ATE(m)"] != full["ATE(m)"]:
            raise RuntimeError(f"{seq}: online/full tracking mismatch in same-run benchmark")

        online_rows.append(online)
        full_rows.append(full)
        details[seq] = {"online": online, "full30k": full, "common": common}

    write_csv(args.result_root / "summary_final_online.csv", online_rows)
    write_csv(args.result_root / "summary_final_full30k.csv", full_rows)
    with (args.result_root / "summary_final_details.json").open("w") as f:
        json.dump(details, f, indent=2)

    print_table("ONLINE SNAPSHOT / same run", online_rows)
    print_table("FULL 30K SNAPSHOT / same run", full_rows)
    print("\nLPIPS: AlexNet LPIPS; lower is better.")
    print("Online Time excludes PSNR/SSIM/LPIPS evaluation.")
    print("Offline Opt is mapper trainForOneIteration time after the online boundary; it excludes metric rendering and final PLY/log saving.")
    print("FULL30K FPS is '-' by design; the online FPS remains reported in the ONLINE table.")


if __name__ == "__main__":
    main()
