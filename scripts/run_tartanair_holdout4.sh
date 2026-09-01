#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${TARTANAIR_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/tartanair_holdout_5_4}"
FORCE="${FORCE:-0}"

if [[ "$#" -gt 0 ]]; then
    SEQUENCES=("$@")
else
    SEQUENCES=(SH000 SH001 SH002 SH003)
fi

BIN="${REPO_ROOT}/bin/tartanair_stereo_eval"
VOCAB="${REPO_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${REPO_ROOT}/cfg/ORB_SLAM3/Stereo/TartanAir/TartanAir.yaml"
BASE_GAUS_CFG="${REPO_ROOT}/cfg/gaussian_mapper/Stereo/TartanAir/TartanAir.yaml"

for p in "$BIN" "$VOCAB" "$ORB_CFG" "$BASE_GAUS_CFG"; do
    if [[ ! -e "$p" ]]; then
        echo "ERROR: required file missing: $p" >&2
        if [[ "$p" == "$BIN" ]]; then
            echo "Build it first: cmake --build build --target tartanair_stereo_eval -j8" >&2
        fi
        exit 2
    fi
done

mkdir -p "$RESULT_ROOT"
TMP_ROOT="/tmp/segsslam_tartanair_holdout_${USER:-user}"
CFG_ROOT="${TMP_ROOT}/cfg"
mkdir -p "$CFG_ROOT"
ONLINE_CFG="${CFG_ROOT}/TartanAir_online.yaml"
FULL_CFG="${CFG_ROOT}/TartanAir_full30k.yaml"

python - "$BASE_GAUS_CFG" "$ONLINE_CFG" "$FULL_CFG" <<'PY'
from pathlib import Path
import re
import sys

src = Path(sys.argv[1]).read_text()

def make(light):
    line = f"Mapper.light_mode: {1 if light else 0}"
    if re.search(r"(?m)^Mapper\.light_mode\s*:.*$", src):
        return re.sub(r"(?m)^Mapper\.light_mode\s*:.*$", line, src)
    marker = "Mapper.use_frequency_regularization:"
    if marker in src:
        return src.replace(marker, line + "\n" + marker, 1)
    return src.rstrip() + "\n\n" + line + "\n"

Path(sys.argv[2]).write_text(make(True))
Path(sys.argv[3]).write_text(make(False))
print("Prepared evaluation configs:")
print(" online :", sys.argv[2])
print(" full30k:", sys.argv[3])
PY

prepare_adapter() {
    local seq="$1"
    local seq_dir="${DATA_ROOT}/stereo/${seq}"
    local left_dir="${seq_dir}/image_left"
    local right_dir="${seq_dir}/image_right"
    local adapter="${TMP_ROOT}/${seq}"

    if [[ ! -d "$left_dir" || ! -d "$right_dir" ]]; then
        echo "ERROR: dataset not found for ${seq}: ${seq_dir}" >&2
        return 3
    fi

    python - "$left_dir" "$right_dir" "$adapter" <<'PY'
import os
import shutil
import sys
from pathlib import Path

left_dir = Path(sys.argv[1])
right_dir = Path(sys.argv[2])
adapter = Path(sys.argv[3])

lefts = sorted(left_dir.glob("*_left.png"))
if not lefts:
    raise SystemExit(f"No left images in {left_dir}")

pairs = []
for left in lefts:
    right = right_dir / left.name.replace("_left.png", "_right.png")
    if not right.is_file():
        raise SystemExit(f"Missing right image: {right}")
    pairs.append((left.resolve(), right.resolve()))

if adapter.exists():
    shutil.rmtree(adapter)
cam0 = adapter / "mav0" / "cam0" / "data"
cam1 = adapter / "mav0" / "cam1" / "data"
cam0.mkdir(parents=True)
cam1.mkdir(parents=True)

period_ns = 100_000_000  # 10 Hz, same adapter convention used in the existing TartanAir runner.
with (adapter / "timestamps.txt").open("w") as f:
    for i, (left, right) in enumerate(pairs):
        stamp = f"{i * period_ns:019d}"
        os.symlink(left, cam0 / f"{stamp}.png")
        os.symlink(right, cam1 / f"{stamp}.png")
        f.write(stamp + "\n")

print(f"Prepared {len(pairs)} frames -> {adapter}")
PY

    printf '%s\n' "$adapter"
}

run_pass() {
    local seq="$1"
    local adapter="$2"
    local mode="$3"
    local cfg="$4"
    local pace="$5"
    local render="$6"
    local out="${RESULT_ROOT}/${seq}/${mode}"

    if [[ "$FORCE" != "1" && -f "${out}/run_stats.json" ]]; then
        echo "[skip] ${seq}/${mode}: run_stats.json already exists (set FORCE=1 to rerun)"
        return 0
    fi

    rm -rf "$out"
    mkdir -p "$out"

    echo
    echo "================================================================"
    echo "RUN ${seq} / ${mode}"
    echo "  pacing : ${pace}"
    echo "  render : ${render}"
    echo "  output : ${out}"
    echo "================================================================"

    "$BIN" \
        "$VOCAB" \
        "$ORB_CFG" \
        "$cfg" \
        "$adapter" \
        "$adapter/timestamps.txt" \
        "$out" \
        "$seq" \
        5 \
        4 \
        "$pace" \
        "$render" \
        2>&1 | tee "$out/run.log"
}

for seq in "${SEQUENCES[@]}"; do
    echo
    echo "######################## ${seq} ########################"
    adapter="$(prepare_adapter "$seq" | tail -n 1)"

    # 1) Online-quality pass: original stream pacing + built-in light mode.
    #    Test frames only estimate pose and never create keyframes.
    run_pass "$seq" "$adapter" online "$ONLINE_CFG" realtime render

    # 2) Complete-quality pass: same 8:2 split and stream, then official 30K optimization.
    run_pass "$seq" "$adapter" full30k "$FULL_CFG" realtime render

    # 3) Dedicated speed pass: compute-limited, no final rendering, light mode.
    #    This keeps metric rendering and 30K optimization out of the FPS number.
    run_pass "$seq" "$adapter" fps "$ONLINE_CFG" compute no_render
done

python "${SCRIPT_DIR}/summarize_tartanair_holdout.py" \
    --result-root "$RESULT_ROOT" \
    --data-root "$DATA_ROOT" \
    --sequences "${SEQUENCES[@]}"

echo
echo "Done."
echo "Online summary : ${RESULT_ROOT}/summary_online.csv"
echo "Full-30K summary: ${RESULT_ROOT}/summary_full30k.csv"
echo "Details        : ${RESULT_ROOT}/summary_details.json"
