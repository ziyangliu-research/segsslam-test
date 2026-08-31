#!/usr/bin/env bash
set -euo pipefail

# TartanAir v1 Stereo Challenge -> SEGS-SLAM adapter.
#
# Usage:
#   bash scripts/tartanair_stereo.sh [SEQUENCE] [MAX_FRAMES] [OUTPUT_DIR]
#
# Examples:
#   bash scripts/tartanair_stereo.sh SE000 200
#   bash scripts/tartanair_stereo.sh SH003 0
#
# MAX_FRAMES=0 means use the full sequence.
# No image data is copied. A temporary EuRoC-style directory containing only
# symlinks and a synthetic 10 Hz timestamp file is generated under /tmp.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQ="${1:-SE000}"
MAX_FRAMES="${2:-0}"

DATA_ROOT="${TARTANAIR_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
SEQ_DIR="${DATA_ROOT}/stereo/${SEQ}"
LEFT_DIR="${SEQ_DIR}/image_left"
RIGHT_DIR="${SEQ_DIR}/image_right"
GT_FILE="${DATA_ROOT}/ground_truth/stereo_gt/${SEQ}.txt"

if [[ ! "${SEQ}" =~ ^(SE|SH)00[0-7]$ ]]; then
    echo "ERROR: sequence must be one of SE000-SE007 or SH000-SH007, got: ${SEQ}" >&2
    exit 2
fi

if [[ ! "${MAX_FRAMES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_FRAMES must be a non-negative integer, got: ${MAX_FRAMES}" >&2
    exit 2
fi

if [[ ! -d "${LEFT_DIR}" ]]; then
    echo "ERROR: left image directory not found: ${LEFT_DIR}" >&2
    exit 3
fi
if [[ ! -d "${RIGHT_DIR}" ]]; then
    echo "ERROR: right image directory not found: ${RIGHT_DIR}" >&2
    exit 3
fi

if [[ "${MAX_FRAMES}" -gt 0 ]]; then
    DEFAULT_OUTPUT="${REPO_ROOT}/results/tartanair_stereo/${SEQ}_first${MAX_FRAMES}"
    ADAPTER_TAG="${SEQ}_first${MAX_FRAMES}"
else
    DEFAULT_OUTPUT="${REPO_ROOT}/results/tartanair_stereo/${SEQ}"
    ADAPTER_TAG="${SEQ}_full"
fi
OUTPUT_DIR="${3:-${DEFAULT_OUTPUT}}"
mkdir -p "${OUTPUT_DIR}"

ADAPTER_ROOT="/tmp/segsslam_tartanair_${USER:-user}/${ADAPTER_TAG}"
TIMESTAMPS_FILE="${ADAPTER_ROOT}/timestamps.txt"

python3 - "${LEFT_DIR}" "${RIGHT_DIR}" "${ADAPTER_ROOT}" "${MAX_FRAMES}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

left_dir = Path(sys.argv[1])
right_dir = Path(sys.argv[2])
adapter_root = Path(sys.argv[3])
max_frames = int(sys.argv[4])

left_images = sorted(left_dir.glob("*_left.png"))
if max_frames > 0:
    left_images = left_images[:max_frames]

if not left_images:
    raise SystemExit(f"ERROR: no *_left.png images found in {left_dir}")

pairs = []
for left in left_images:
    right_name = left.name.replace("_left.png", "_right.png")
    right = right_dir / right_name
    if not right.is_file():
        raise SystemExit(f"ERROR: missing right image for {left.name}: {right}")
    pairs.append((left.resolve(), right.resolve()))

if adapter_root.exists():
    shutil.rmtree(adapter_root)

cam0 = adapter_root / "mav0" / "cam0" / "data"
cam1 = adapter_root / "mav0" / "cam1" / "data"
cam0.mkdir(parents=True)
cam1.mkdir(parents=True)

# The challenge package has frame indices but no timestamps.  SEGS-SLAM's
# EuRoC loader only requires a monotonic timestamp list, so synthesize 10 Hz.
period_ns = 100_000_000
with (adapter_root / "timestamps.txt").open("w") as f:
    for i, (left, right) in enumerate(pairs):
        stamp = i * period_ns
        stamp_text = f"{stamp:019d}"
        os.symlink(left, cam0 / f"{stamp_text}.png")
        os.symlink(right, cam1 / f"{stamp_text}.png")
        f.write(stamp_text + "\n")

print(f"Prepared {len(pairs)} stereo pairs")
print(f"Adapter: {adapter_root}")
print(f"First left:  {pairs[0][0]}")
print(f"First right: {pairs[0][1]}")
print(f"Last left:   {pairs[-1][0]}")
PY

BIN="${REPO_ROOT}/bin/euroc_stereo"
VOCAB="${REPO_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${REPO_ROOT}/cfg/ORB_SLAM3/Stereo/TartanAir/TartanAir.yaml"
GAUS_CFG="${REPO_ROOT}/cfg/gaussian_mapper/Stereo/TartanAir/TartanAir.yaml"

if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: executable not found: ${BIN}" >&2
    echo "Build SEGS-SLAM first, then rerun this script." >&2
    exit 4
fi
if [[ ! -f "${VOCAB}" ]]; then
    echo "ERROR: ORB vocabulary not found: ${VOCAB}" >&2
    exit 4
fi

cat <<EOF

============================================================
SEGS-SLAM TartanAir stereo run
============================================================
Sequence : ${SEQ}
Images   : ${SEQ_DIR}
GT pose  : ${GT_FILE}  (not used yet; reserved for later ATE evaluation)
Output   : ${OUTPUT_DIR}
ORB cfg  : ${ORB_CFG}
GS cfg   : ${GAUS_CFG}
Viewer   : disabled
============================================================

EOF

cd "${REPO_ROOT}"

"${BIN}" \
    "${VOCAB}" \
    "${ORB_CFG}" \
    "${GAUS_CFG}" \
    "${ADAPTER_ROOT}" \
    "${TIMESTAMPS_FILE}" \
    "${OUTPUT_DIR}" \
    no_viewer
