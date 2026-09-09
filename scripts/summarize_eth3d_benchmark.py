#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_csv(path: Path):
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


def interp_gt_position(t, gt_t, gt_xyz, max_gap):
    j = int(np.searchsorted(gt_t, t))
    if j == 0:
        if abs(gt_t[0] - t) <= max_gap:
            return gt_xyz[0]
        return None
    if j >= len(gt_t):
        if abs(t - gt_t[-1]) <= max_gap:
            return gt_xyz[-1]
        return None
    t0, t1 = gt_t[j - 1], gt_t[j]
    if (t1 - t0) > max_gap:
        return None
    if t1 <= t0:
        return None
    a = (t - t0) / (t1 - t0)
    return (1.0 - a) * gt_xyz[j - 1] + a * gt_xyz[j]


def compute_ate(eval_csv: Path, manifest_csv: Path, gt_file: Path, max_gt_gap):
    rows = load_csv(eval_csv)
    manifest_rows = load_csv(manifest_csv)
    manifest = {int(r["frame"]): float(r["source_timestamp"]) for r in manifest_rows}

    gt = np.loadtxt(gt_file, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt[None, :]
    gt = gt[np.argsort(gt[:, 0])]
    gt_t = gt[:, 0]
    gt_xyz = gt[:, 1:4]

    est, target = [], []
    skipped_gap = 0
    for row in rows:
        if row["tracked"] != "1" or row["largest_map"] != "1":
            continue
        frame = int(row["frame"])
        if frame not in manifest:
            continue
        xyz = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
        if not np.isfinite(xyz).all():
            continue
        g = interp_gt_position(manifest[frame], gt_t, gt_xyz, max_gt_gap)
        if g is None:
            skipped_gap += 1
            continue
        est.append(xyz)
        target.append(g)

    if len(est) < 3:
        return float("nan"), len(est), skipped_gap
    est = np.asarray(est)
    target = np.asarray(target)
    aligned = rigid_align_se3(est, target)
    err = np.linalg.norm(aligned - target, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), len(est), skipped_gap


def fields():
    return [
        "Sequence", "MaxMap", "Train PSNR/SSIM/LPIPS", "Test PSNR/SSIM/LPIPS",
        "ATE(m)", "FPS", "Online Time(s)", "Offline Opt(s)", "Total Time(s)", "Gaussians"
    ]


def missing_row(seq):
    return {k: (seq if k == "Sequence" else "MISSING") for k in fields()}


def collect(seq, mode, root: Path, seq_data: Path, max_gt_gap):
    mode_dir = root / mode
    common = load_json(root / "common_stats.json")
    stats = load_json(mode_dir / "run_stats.json")
    render = load_json(mode_dir / "render_summary.json")
    if "train_lpips" not in render or "test_lpips" not in render:
        raise RuntimeError(f"LPIPS missing in {mode_dir / 'render_summary.json'}")
    ate, ate_frames, skipped_gap = compute_ate(
        mode_dir / "per_frame_eval.csv",
        root / "frame_manifest.csv",
        seq_data / "groundtruth_left.txt",
        max_gt_gap,
    )
    online_t = float(common["online_total_seconds"])
    offline_t = float(common["offline_optimization_seconds"])
    total_t = online_t if mode == "online" else float(common["total_compute_seconds"])
    row = {
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
    }
    detail = {"ate_frames": ate_frames, "gt_gap_skipped_frames": skipped_gap}
    return row, detail


def write_csv(path, rows):
    fs = fields()
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fs)
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--sequences", nargs="+", default=["mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3"])
    ap.add_argument("--max-gt-gap-sec", type=float, default=0.1)
    args = ap.parse_args()

    online_rows, full_rows, details = [], [], {}
    for seq in args.sequences:
        root = args.result_root / seq / "benchmark"
        seq_data = args.data_root / seq
        try:
            online, od = collect(seq, "online", root, seq_data, args.max_gt_gap_sec)
        except Exception as e:
            print(f"[MISSING online] {seq}: {e}")
            online, od = missing_row(seq), {"error": str(e)}
        try:
            full, fd = collect(seq, "full30k", root, seq_data, args.max_gt_gap_sec)
        except Exception as e:
            print(f"[MISSING full30k] {seq}: {e}")
            full, fd = missing_row(seq), {"error": str(e)}

        if online["MaxMap"] != "MISSING" and full["MaxMap"] != "MISSING":
            if online["MaxMap"] != full["MaxMap"] or online["ATE(m)"] != full["ATE(m)"]:
                raise RuntimeError(f"{seq}: same-run online/full tracking mismatch")
        online_rows.append(online)
        full_rows.append(full)
        details[seq] = {"online": od, "full30k": fd}

    args.result_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.result_root / "summary_eth3d_online.csv", online_rows)
    write_csv(args.result_root / "summary_eth3d_full30k.csv", full_rows)
    (args.result_root / "summary_eth3d_details.json").write_text(json.dumps(details, indent=2) + "\n")
    print_table("ETH3D ONLINE SNAPSHOT", online_rows)
    print_table("ETH3D FULL 30K SNAPSHOT", full_rows)
    print(f"\nATE uses timestamp association with linear GT translation interpolation; GT gaps > {args.max_gt_gap_sec:.3f}s are skipped.")
    print("LPIPS: AlexNet LPIPS; lower is better.")


if __name__ == "__main__":
    main()
