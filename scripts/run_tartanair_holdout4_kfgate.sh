#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SEGS_HOLDOUT_PERIOD=5
export SEGS_HOLDOUT_OFFSET=4

if ! grep -q "suppressMappingForHoldoutFrame" "${REPO_ROOT}/ORB-SLAM3/src/Tracking.cc"; then
    echo "ERROR: holdout keyframe gate has not been applied to ORB-SLAM3/src/Tracking.cc" >&2
    echo "Run:" >&2
    echo "  python scripts/apply_tartanair_holdout_gate.py" >&2
    echo "  cmake --build ORB-SLAM3/build -j8" >&2
    exit 2
fi

cat <<EOF
============================================================
TartanAir holdout_5_4 benchmark
Tracking path : original ORB-SLAM3 SLAM tracking
TEST frames   : frame index 4,9,14,...
TEST mapping  : keyframe/map insertion suppressed only
Gate env      : period=${SEGS_HOLDOUT_PERIOD}, offset=${SEGS_HOLDOUT_OFFSET}
============================================================
EOF

exec bash "${SCRIPT_DIR}/run_tartanair_holdout4.sh" "$@"
