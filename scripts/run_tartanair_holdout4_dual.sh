#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${TARTANAIR_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/tartanair_holdout_5_4}"
FORCE_QUALITY="${FORCE_QUALITY:-0}"
RERUN_FPS="${RERUN_FPS:-0}"

if [[ "$#" -gt 0 ]]; then
    SEQUENCES=("$@")
else
    SEQUENCES=(SH000 SH001 SH002 SH003)
fi

DUAL_BIN="${REPO_ROOT}/bin/tartanair_stereo_dual_eval"
FPS_BIN="${REPO_ROOT}/bin/tartanair_stereo_eval"
VOCAB="${REPO_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${REPO_ROOT}/cfg/ORB_SLAM3/Stereo/TartanAir/TartanAir.yaml"
BASE_GAUS_CFG="${REPO_ROOT}/cfg/gaussian_mapper/Stereo/TartanAir/TartanAir.yaml"

for p in "$DUAL_BIN" "$FPS_BIN" "$VOCAB" "$ORB_CFG" "$BASE_GAUS_CFG"; do
    [[ -e "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done

mkdir -p "$RESULT_ROOT"
TMP_ROOT="/tmp/segsslam_tartanair_dual_${USER:-user}"
CFG_ROOT="${TMP_ROOT}/cfg"
mkdir -p "$CFG_ROOT"
FULL_CFG="${CFG_ROOT}/TartanAir_full30k.yaml"
ONLINE_CFG="${CFG_ROOT}/TartanAir_online_light.yaml"

python - "$BASE_GAUS_CFG" "$FULL_CFG" "$ONLINE_CFG" <<'PY'
from pathlib import Path
import re, sys
src = Path(sys.argv[1]).read_text()
def set_light(v):
    line = f"Mapper.light_mode: {v}"
    if re.search(r"(?m)^Mapper\.light_mode\s*:.*$", src):
        return re.sub(r"(?m)^Mapper\.light_mode\s*:.*$", line, src)
    return src.rstrip() + "\n" + line + "\n"
Path(sys.argv[2]).write_text(set_light(0))
Path(sys.argv[3]).write_text(set_light(1))
PY

prepare_adapter() {
    local seq="$1"
    local left_dir="${DATA_ROOT}/stereo/${seq}/image_left"
    local right_dir="${DATA_ROOT}/stereo/${seq}/image_right"
    local adapter="${TMP_ROOT}/${seq}"
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
print(adapter)
PY
}

export SEGS_HOLDOUT_PERIOD=5
export SEGS_HOLDOUT_OFFSET=4

for seq in "${SEQUENCES[@]}"; do
    echo
    echo "######################## ${seq} ########################"
    adapter="$(prepare_adapter "$seq" | tail -n 1)"
    dual_out="${RESULT_ROOT}/${seq}/dual"
    fps_out="${RESULT_ROOT}/${seq}/fps"

    if [[ "$FORCE_QUALITY" != "1" && -f "${dual_out}/online/run_stats.json" && -f "${dual_out}/full30k/run_stats.json" ]]; then
        echo "[skip] ${seq}/dual quality snapshots already exist"
    else
        rm -rf "$dual_out"
        mkdir -p "$dual_out"
        echo "[run] ${seq}: ONE tracking run -> ONLINE snapshot -> same mapper continues to FULL30K"
        "$DUAL_BIN" \
            "$VOCAB" \
            "$ORB_CFG" \
            "$FULL_CFG" \
            "$adapter" \
            "$adapter/timestamps.txt" \
            "$dual_out" \
            "$seq" \
            5 4 realtime render \
            2>&1 | tee "$dual_out/run.log"
    fi

    if [[ "$RERUN_FPS" != "1" && -f "${fps_out}/run_stats.json" ]]; then
        echo "[reuse] ${seq}/fps from previous compute-limited run"
    else
        rm -rf "$fps_out"
        mkdir -p "$fps_out"
        echo "[run] ${seq}: FPS pass (only because no reusable FPS result exists)"
        "$FPS_BIN" \
            "$VOCAB" \
            "$ORB_CFG" \
            "$ONLINE_CFG" \
            "$adapter" \
            "$adapter/timestamps.txt" \
            "$fps_out" \
            "$seq" \
            5 4 compute no_render \
            2>&1 | tee "$fps_out/run.log"
    fi
done

python "${SCRIPT_DIR}/summarize_tartanair_dual.py" \
    --result-root "$RESULT_ROOT" \
    --data-root "$DATA_ROOT" \
    --sequences "${SEQUENCES[@]}"

echo
echo "Done."
echo "Online same-run : ${RESULT_ROOT}/summary_dual_online.csv"
echo "Full30K same-run: ${RESULT_ROOT}/summary_dual_full30k.csv"
