#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${ETH3D_RECT_ROOT:-/home/shiyo/Desktop/Datasets/ETH3D_rectified}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/eth3d_holdout_5_4_final4}"
JOBS="${JOBS:-8}"
LPIPS_BATCH="${LPIPS_BATCH:-8}"
FORCE="${FORCE:-0}"
MAX_GT_GAP_SEC="${MAX_GT_GAP_SEC:-0.1}"

if [[ "$#" -gt 0 ]]; then
    SEQUENCES=("$@")
else
    SEQUENCES=(mannequin_face_1 einstein_1 sofa_3 plant_scene_3)
fi

cd "$REPO_ROOT"
if [[ -f "$REPO_ROOT/env_sm120.sh" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env_sm120.sh"
fi

python "$SCRIPT_DIR/compute_eth3d_lpips.py" --preflight

# Reuse the same evaluation-only holdout and same-run snapshot instrumentation
# already used for the TartanAir benchmark.
python "$SCRIPT_DIR/apply_tartanair_holdout_gate.py"
python "$SCRIPT_DIR/apply_dual_snapshot_barrier.py"
python "$SCRIPT_DIR/apply_tartanair_benchmark_target.py"

cmake --build "$REPO_ROOT/ORB-SLAM3/build" -j"$JOBS"
cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" \
    -DOpenCV_DIR="$OpenCV_DIR" \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$REPO_ROOT/build" --target tartanair_stereo_benchmark -j"$JOBS"

BIN="$REPO_ROOT/bin/tartanair_stereo_benchmark"
VOCAB="$REPO_ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_TEMPLATE="$REPO_ROOT/cfg/ORB_SLAM3/Stereo/TartanAir/TartanAir.yaml"
GAUS_TEMPLATE="$REPO_ROOT/cfg/gaussian_mapper/Stereo/TartanAir/TartanAir.yaml"
for p in "$BIN" "$VOCAB" "$ORB_TEMPLATE" "$GAUS_TEMPLATE"; do
    [[ -e "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done

mkdir -p "$RESULT_ROOT"
TMP_ROOT="/tmp/segsslam_eth3d_final4_${USER:-user}"
mkdir -p "$TMP_ROOT"

export SEGS_HOLDOUT_PERIOD=5
export SEGS_HOLDOUT_OFFSET=4

for seq in "${SEQUENCES[@]}"; do
    echo
    echo "################################################################"
    echo "### ETH3D: $seq"
    echo "################################################################"

    seq_data="$DATA_ROOT/$seq"
    adapter="$TMP_ROOT/$seq/adapter"
    cfg_dir="$TMP_ROOT/$seq/cfg"
    mkdir -p "$cfg_dir"

    python "$SCRIPT_DIR/prepare_eth3d_benchmark.py" \
        --sequence-dir "$seq_data" \
        --adapter-dir "$adapter" \
        --cfg-dir "$cfg_dir" \
        --orb-template "$ORB_TEMPLATE" \
        --gaussian-template "$GAUS_TEMPLATE"

    out="$RESULT_ROOT/$seq/benchmark"
    mkdir -p "$out"
    cp "$adapter/frame_manifest.csv" "$out/frame_manifest.csv"
    cp "$adapter/adapter_meta.json" "$out/adapter_meta.json"
    cp "$cfg_dir/ORB_ETH3D.yaml" "$out/ORB_ETH3D.yaml"
    cp "$cfg_dir/Gaussian_ETH3D_full30k.yaml" "$out/Gaussian_ETH3D_full30k.yaml"

    if [[ "$FORCE" != "1" && -f "$out/.quality_complete" ]]; then
        echo "[skip quality] $seq already completed; reusing snapshots"
    else
        rm -rf "$out/online" "$out/full30k" "$out/run.log" "$out/.quality_complete" "$out/common_stats.json"
        echo "[run] $seq: ONE compute-limited run -> ONLINE snapshot -> same mapper -> FULL30K"
        "$BIN" \
            "$VOCAB" \
            "$cfg_dir/ORB_ETH3D.yaml" \
            "$cfg_dir/Gaussian_ETH3D_full30k.yaml" \
            "$adapter" \
            "$adapter/timestamps.txt" \
            "$out" \
            "$seq" \
            5 4 \
            2>&1 | tee "$out/run.log"
    fi

    for mode in online full30k; do
        if [[ "$FORCE" != "1" && -f "$out/$mode/lpips_summary.json" && -d "$out/$mode/test_renders" ]]; then
            echo "[skip LPIPS/test renders] $seq/$mode already exists"
        else
            echo "[LPIPS + test render export] $seq/$mode"
            python "$SCRIPT_DIR/compute_eth3d_lpips.py" \
                --snapshot-dir "$out/$mode" \
                --manifest "$out/frame_manifest.csv" \
                --batch-size "$LPIPS_BATCH" \
                --save-test-renders
        fi
    done

done

python "$SCRIPT_DIR/summarize_eth3d_benchmark.py" \
    --result-root "$RESULT_ROOT" \
    --data-root "$DATA_ROOT" \
    --sequences "${SEQUENCES[@]}" \
    --max-gt-gap-sec "$MAX_GT_GAP_SEC"

echo
echo "Done."
echo "ONLINE : $RESULT_ROOT/summary_eth3d_online.csv"
echo "FULL30K: $RESULT_ROOT/summary_eth3d_full30k.csv"
echo "Test renders are under each sequence's benchmark/{online,full30k}/test_renders/"
