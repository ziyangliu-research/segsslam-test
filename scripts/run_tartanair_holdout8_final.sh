#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${TARTANAIR_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/tartanair_holdout_5_4_final8}"
JOBS="${JOBS:-8}"
LPIPS_BATCH="${LPIPS_BATCH:-8}"
FORCE="${FORCE:-0}"

if [[ "$#" -gt 0 ]]; then
    SEQUENCES=("$@")
else
    SEQUENCES=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)
fi

cd "$REPO_ROOT"
if [[ -f "$REPO_ROOT/env_sm120.sh" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env_sm120.sh"
fi

# Fail before any long experiment if LPIPS or its AlexNet weights are unavailable.
python "$SCRIPT_DIR/compute_tartanair_lpips.py" --preflight

# Evaluation-only patches are idempotent.
python "$SCRIPT_DIR/apply_tartanair_holdout_gate.py"
python "$SCRIPT_DIR/apply_dual_snapshot_barrier.py"
python "$SCRIPT_DIR/apply_tartanair_benchmark_target.py"

# Rebuild ORB-SLAM3 only if the holdout gate changed Tracking.cc; otherwise this is a no-op.
cmake --build "$REPO_ROOT/ORB-SLAM3/build" -j"$JOBS"

# Reconfigure the root build so newly added benchmark targets are visible.
cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" \
    -DOpenCV_DIR="$OpenCV_DIR" \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$REPO_ROOT/build" --target tartanair_stereo_benchmark -j"$JOBS"

BIN="$REPO_ROOT/bin/tartanair_stereo_benchmark"
VOCAB="$REPO_ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="$REPO_ROOT/cfg/ORB_SLAM3/Stereo/TartanAir/TartanAir.yaml"
BASE_GAUS_CFG="$REPO_ROOT/cfg/gaussian_mapper/Stereo/TartanAir/TartanAir.yaml"
for p in "$BIN" "$VOCAB" "$ORB_CFG" "$BASE_GAUS_CFG"; do
    [[ -e "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done

mkdir -p "$RESULT_ROOT"
TMP_ROOT="/tmp/segsslam_tartanair_final8_${USER:-user}"
CFG_ROOT="$TMP_ROOT/cfg"
mkdir -p "$CFG_ROOT"
FULL_CFG="$CFG_ROOT/TartanAir_full30k.yaml"

python - "$BASE_GAUS_CFG" "$FULL_CFG" <<'PY'
from pathlib import Path
import re, sys
src = Path(sys.argv[1]).read_text()
line = "Mapper.light_mode: 0"
if re.search(r"(?m)^Mapper\.light_mode\s*:.*$", src):
    src = re.sub(r"(?m)^Mapper\.light_mode\s*:.*$", line, src)
else:
    src = src.rstrip() + "\n" + line + "\n"
Path(sys.argv[2]).write_text(src)
PY

prepare_adapter() {
    local seq="$1"
    local left_dir="$DATA_ROOT/stereo/$seq/image_left"
    local right_dir="$DATA_ROOT/stereo/$seq/image_right"
    local adapter="$TMP_ROOT/$seq"
    python - "$left_dir" "$right_dir" "$adapter" <<'PY'
import os, shutil, sys
from pathlib import Path
left_dir, right_dir, adapter = map(Path, sys.argv[1:])
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
cam0 = adapter / "mav0/cam0/data"
cam1 = adapter / "mav0/cam1/data"
cam0.mkdir(parents=True)
cam1.mkdir(parents=True)
with (adapter / "timestamps.txt").open("w") as f:
    for i, (left, right) in enumerate(pairs):
        stamp = f"{i * 100_000_000:019d}"
        os.symlink(left, cam0 / f"{stamp}.png")
        os.symlink(right, cam1 / f"{stamp}.png")
        f.write(stamp + "\n")
print(f"Prepared {len(pairs)} frames for {left_dir.parent.name}", file=sys.stderr)
print(adapter)
PY
}

export SEGS_HOLDOUT_PERIOD=5
export SEGS_HOLDOUT_OFFSET=4

for seq in "${SEQUENCES[@]}"; do
    echo
    echo "################################################################"
    echo "### $seq"
    echo "################################################################"
    adapter="$(prepare_adapter "$seq" | tail -n 1)"
    out="$RESULT_ROOT/$seq/benchmark"

    if [[ "$FORCE" != "1" && -f "$out/.quality_complete" ]]; then
        echo "[skip quality] $seq already completed; reusing snapshots"
    else
        rm -rf "$out"
        mkdir -p "$out"
        echo "[run] $seq: ONE compute-limited run -> ONLINE snapshot -> same mapper -> FULL30K"
        "$BIN" \
            "$VOCAB" \
            "$ORB_CFG" \
            "$FULL_CFG" \
            "$adapter" \
            "$adapter/timestamps.txt" \
            "$out" \
            "$seq" \
            5 4 \
            2>&1 | tee "$out/run.log"
    fi

    for mode in online full30k; do
        if [[ "$FORCE" != "1" && -f "$out/$mode/lpips_summary.json" ]]; then
            echo "[skip LPIPS] $seq/$mode already exists"
        else
            echo "[LPIPS] $seq/$mode"
            python "$SCRIPT_DIR/compute_tartanair_lpips.py" \
                --snapshot-dir "$out/$mode" \
                --data-root "$DATA_ROOT" \
                --sequence "$seq" \
                --batch-size "$LPIPS_BATCH"
        fi
    done

done

python "$SCRIPT_DIR/summarize_tartanair_benchmark.py" \
    --result-root "$RESULT_ROOT" \
    --data-root "$DATA_ROOT" \
    --sequences "${SEQUENCES[@]}"

echo
echo "Done."
echo "ONLINE : $RESULT_ROOT/summary_final_online.csv"
echo "FULL30K: $RESULT_ROOT/summary_final_full30k.csv"
