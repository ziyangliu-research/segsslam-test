#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np


def timestamp_to_ns(text: str):
    v = Decimal(text)
    # ETH3D RGB names are normally timestamps in seconds. Also tolerate an
    # already-nanosecond integer representation.
    if abs(v) >= Decimal("1e12") and v == v.to_integral_value():
        ns = int(v)
        sec = float(v / Decimal("1e9"))
    else:
        ns = int((v * Decimal("1e9")).to_integral_value(rounding=ROUND_HALF_UP))
        sec = float(v)
    return ns, sec


def replace_scalar(text, key, value):
    pat = rf"(?m)^{re.escape(key)}\s*:.*$"
    line = f"{key}: {value}"
    if not re.search(pat, text):
        raise RuntimeError(f"Missing YAML key in template: {key}")
    return re.sub(pat, line, text, count=1)


def replace_stereo_matrix(text, baseline):
    pat = re.compile(
        r"(?ms)^Stereo\.T_c1_c2:\s*!!opencv-matrix\s*\n"
        r"\s*rows:\s*4\s*\n\s*cols:\s*4\s*\n\s*dt:\s*f\s*\n"
        r"\s*data:\s*\[[^\]]*\]"
    )
    block = (
        "Stereo.T_c1_c2: !!opencv-matrix\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  dt: f\n"
        f"  data: [1.0, 0.0, 0.0, {baseline:.9f},\n"
        "         0.0, 1.0, 0.0, 0.0,\n"
        "         0.0, 0.0, 1.0, 0.0,\n"
        "         0.0, 0.0, 0.0, 1.0]"
    )
    if not pat.search(text):
        raise RuntimeError("Stereo.T_c1_c2 block not found in ORB template")
    return pat.sub(block, text, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--orb-template", type=Path, required=True)
    ap.add_argument("--gaussian-template", type=Path, required=True)
    args = ap.parse_args()

    seq = args.sequence_dir.resolve()
    left_dir = seq / "image_left"
    right_dir = seq / "image_right"
    calib_path = seq / "calibration.json"
    timestamp_path = seq / "timestamps.txt"
    gt_path = seq / "groundtruth_left.txt"
    for p in (left_dir, right_dir, calib_path, timestamp_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    calib = json.loads(calib_path.read_text())
    K1 = np.asarray(calib["K_rectified_left"], dtype=float)
    K2 = np.asarray(calib["K_rectified_right"], dtype=float)
    if not np.allclose(K1, K2, atol=1e-5, rtol=1e-7):
        raise RuntimeError(f"Rectified K mismatch in {calib_path}")
    width, height = map(int, calib["rectified_size"])
    baseline = float(calib["signed_baseline_rectified_m"])
    if abs(baseline) <= 1e-6:
        raise RuntimeError("Invalid rectified baseline")

    stamps = [x.strip() for x in timestamp_path.read_text().splitlines() if x.strip() and not x.startswith("#")]
    if not stamps:
        raise RuntimeError(f"No timestamps in {timestamp_path}")

    # Resolve source images by exact timestamp stem. This preserves the
    # rectification script's ordering and avoids accidental lexical reordering.
    pairs = []
    secs = []
    used_ns = set()
    for frame, stamp in enumerate(stamps):
        left = left_dir / f"{stamp}.png"
        right = right_dir / f"{stamp}.png"
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"Missing rectified pair for timestamp {stamp}: {left} / {right}")
        ns, sec = timestamp_to_ns(stamp)
        if ns in used_ns:
            raise RuntimeError(f"Timestamp collision after ns conversion: {stamp} -> {ns}")
        used_ns.add(ns)
        pairs.append((frame, stamp, ns, left.resolve(), right.resolve()))
        secs.append(sec)

    if len(secs) > 1:
        dts = np.diff(np.asarray(secs, dtype=float))
        dts = dts[dts > 1e-6]
        fps = int(round(1.0 / float(np.median(dts)))) if len(dts) else 30
    else:
        fps = 30
    fps = max(1, min(240, fps))

    adapter = args.adapter_dir.resolve()
    if adapter.exists():
        shutil.rmtree(adapter)
    cam0 = adapter / "mav0/cam0/data"
    cam1 = adapter / "mav0/cam1/data"
    cam0.mkdir(parents=True)
    cam1.mkdir(parents=True)

    with (adapter / "timestamps.txt").open("w") as tf, (adapter / "frame_manifest.csv").open("w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["frame", "source_timestamp", "adapter_timestamp_ns", "left_source", "right_source"])
        for frame, stamp, ns, left, right in pairs:
            name = f"{ns:019d}.png"
            os.symlink(left, cam0 / name)
            os.symlink(right, cam1 / name)
            tf.write(f"{ns:019d}\n")
            w.writerow([frame, stamp, ns, str(left), str(right)])

    args.cfg_dir.mkdir(parents=True, exist_ok=True)
    orb = args.orb_template.read_text()
    fx, fy, cx, cy = float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2])
    for prefix in ("Camera1", "Camera2"):
        orb = replace_scalar(orb, f"{prefix}.fx", f"{fx:.9f}")
        orb = replace_scalar(orb, f"{prefix}.fy", f"{fy:.9f}")
        orb = replace_scalar(orb, f"{prefix}.cx", f"{cx:.9f}")
        orb = replace_scalar(orb, f"{prefix}.cy", f"{cy:.9f}")
    orb = replace_scalar(orb, "Camera.width", width)
    orb = replace_scalar(orb, "Camera.height", height)
    orb = replace_scalar(orb, "Camera.fps", fps)
    orb = replace_stereo_matrix(orb, baseline)
    (args.cfg_dir / "ORB_ETH3D.yaml").write_text(orb)

    gaus = args.gaussian_template.read_text()
    light_pat = r"(?m)^Mapper\.light_mode\s*:.*$"
    if re.search(light_pat, gaus):
        gaus = re.sub(light_pat, "Mapper.light_mode: 0", gaus, count=1)
    else:
        gaus = gaus.rstrip() + "\nMapper.light_mode: 0\n"
    (args.cfg_dir / "Gaussian_ETH3D_full30k.yaml").write_text(gaus)

    meta = {
        "sequence": seq.name,
        "frames": len(pairs),
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "signed_baseline_m": baseline,
        "camera_fps_from_median_timestamp_dt": fps,
        "groundtruth_left": str(gt_path.resolve()),
    }
    (adapter / "adapter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
