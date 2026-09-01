#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "gaussian_mapper.h"
SOURCE = ROOT / "src" / "gaussian_mapper.cpp"
CMAKE = ROOT / "CMakeLists.txt"

HEADER_MARK = "waitForEvaluationSnapshotReady"
HEADER_STATE_MARK = "evaluation_snapshot_barrier_enabled_"
SOURCE_MARK = "[dual snapshot] online boundary ready"
TARGET_MARK = "tartanair_stereo_dual_eval"


def patch_header():
    s = HEADER.read_text()

    # Public API. Match structurally instead of depending on exact whitespace.
    if HEADER_MARK not in s:
        pat = re.compile(
            r"(\s*bool\s+isStopped\(\);\s*\n"
            r"\s*void\s+signalStop\(const bool going_to_stop = true\);\s*\n)"
        )
        m = pat.search(s)
        if not m:
            raise SystemExit("ERROR: gaussian_mapper.h public insertion point not found")
        insert = m.group(1) + """
    // Evaluation-only barrier used by the TartanAir dual-snapshot benchmark.
    // It pauses the mapper exactly once after SLAM has shut down and final
    // SLAM poses have been synchronized, before post-stream 30K training.
    void enableEvaluationSnapshotBarrier(const bool enabled = true);
    bool waitForEvaluationSnapshotReady();
    void releaseEvaluationSnapshot();
    int getEvaluationSnapshotIteration();
"""
        s = s[:m.start()] + insert + s[m.end():]

    # Protected synchronization state. Insert immediately after mutex_render_.
    # This deliberately tolerates trailing spaces/tabs and local formatting.
    if HEADER_STATE_MARK not in s:
        pat = re.compile(r"(?m)^(\s*std::mutex\s+mutex_render_;)[ \t]*$")
        m = pat.search(s)
        if not m:
            raise SystemExit("ERROR: gaussian_mapper.h mutex_render_ insertion point not found")
        insert = m.group(1) + """

    // Evaluation-only dual-snapshot synchronization. Disabled by default,
    // therefore normal SEGS-SLAM runs are behavior-identical to upstream.
    bool evaluation_snapshot_barrier_enabled_ = false;
    bool evaluation_snapshot_ready_ = false;
    bool evaluation_snapshot_released_ = false;
    int evaluation_snapshot_iteration_ = -1;
    std::mutex mutex_evaluation_snapshot_;
    std::condition_variable cv_evaluation_snapshot_;"""
        s = s[:m.start()] + insert + s[m.end():]

    HEADER.write_text(s)
    print("[dual-snapshot] patched include/gaussian_mapper.h")


def patch_source():
    s = SOURCE.read_text()

    # Add the public barrier methods after signalStop().
    if "void GaussianMapper::enableEvaluationSnapshotBarrier" not in s:
        pat = re.compile(
            r"void\s+GaussianMapper::signalStop\(const bool going_to_stop\)\s*\n"
            r"\{\s*\n"
            r"\s*std::unique_lock<std::mutex>\s+lock_status\(this->mutex_status_\);\s*\n"
            r"\s*this->stopped_\s*=\s*going_to_stop;\s*\n"
            r"\}\s*\n"
        )
        m = pat.search(s)
        if not m:
            raise SystemExit("ERROR: gaussian_mapper.cpp signalStop insertion point not found")
        methods = m.group(0) + r'''
void GaussianMapper::enableEvaluationSnapshotBarrier(const bool enabled)
{
    std::unique_lock<std::mutex> lock(mutex_evaluation_snapshot_);
    evaluation_snapshot_barrier_enabled_ = enabled;
    evaluation_snapshot_ready_ = false;
    evaluation_snapshot_released_ = false;
    evaluation_snapshot_iteration_ = -1;
}

bool GaussianMapper::waitForEvaluationSnapshotReady()
{
    std::unique_lock<std::mutex> lock(mutex_evaluation_snapshot_);
    cv_evaluation_snapshot_.wait(lock, [this]() {
        return evaluation_snapshot_ready_ || !evaluation_snapshot_barrier_enabled_;
    });
    return evaluation_snapshot_ready_;
}

void GaussianMapper::releaseEvaluationSnapshot()
{
    {
        std::unique_lock<std::mutex> lock(mutex_evaluation_snapshot_);
        evaluation_snapshot_released_ = true;
    }
    cv_evaluation_snapshot_.notify_all();
}

int GaussianMapper::getEvaluationSnapshotIteration()
{
    std::unique_lock<std::mutex> lock(mutex_evaluation_snapshot_);
    return evaluation_snapshot_iteration_;
}

'''
        s = s[:m.start()] + methods + s[m.end():]

    # Pause exactly at the existing post-SLAM boundary, before light/full tail logic.
    if SOURCE_MARK not in s:
        pat = re.compile(r"(?m)^(\s*)if\s*\(light_mode\)\s*\n\s*break;\s*$")
        m = pat.search(s)
        if not m:
            raise SystemExit("ERROR: gaussian_mapper.cpp online boundary insertion point not found")
        indent = m.group(1)
        barrier = f'''{indent}// Evaluation-only pause point. At this point the input stream has ended,
{indent}// ORB-SLAM3 has finished, and final SLAM poses have been synchronized
{indent}// into Gaussian keyframes. No explicit post-stream 30K stage has begun.
{indent}{{
{indent}    std::unique_lock<std::mutex> snapshot_lock(mutex_evaluation_snapshot_);
{indent}    if (evaluation_snapshot_barrier_enabled_) {{
{indent}        evaluation_snapshot_iteration_ = getIteration();
{indent}        evaluation_snapshot_ready_ = true;
{indent}        std::cout << "\\n[dual snapshot] online boundary ready at iteration "
{indent}                  << evaluation_snapshot_iteration_ << std::endl;
{indent}        cv_evaluation_snapshot_.notify_all();
{indent}        cv_evaluation_snapshot_.wait(snapshot_lock, [this]() {{
{indent}            return evaluation_snapshot_released_;
{indent}        }});
{indent}        std::cout << "[dual snapshot] released; continuing original post-stream optimization"
{indent}                  << std::endl;
{indent}    }}
{indent}}}

{m.group(0)}'''
        s = s[:m.start()] + barrier + s[m.end():]

    SOURCE.write_text(s)
    print("[dual-snapshot] patched src/gaussian_mapper.cpp")


def patch_cmake():
    s = CMAKE.read_text()
    if TARGET_MARK in s:
        print("[dual-snapshot] CMake target already present")
        return

    pat = re.compile(
        r"(add_executable\(tartanair_stereo_eval\s+examples/tartanair_stereo_eval\.cpp\)\s*\n"
        r"target_link_libraries\(tartanair_stereo_eval\s*\n"
        r"\s*gaussian_viewer\s*\n"
        r"\s*gaussian_mapper\s*\n"
        r"\s*\$\{ORB_SLAM3_SOURCE_DIR\}/lib/libORB_SLAM3\.so\)\s*\n)"
    )
    m = pat.search(s)
    if not m:
        raise SystemExit("ERROR: CMake tartanair_stereo_eval target not found")

    addition = m.group(1) + """
# TartanAir one-run online + full30K evaluator (evaluation-only barrier)
add_executable(tartanair_stereo_dual_eval examples/tartanair_stereo_dual_eval.cpp)
target_link_libraries(tartanair_stereo_dual_eval
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)
"""
    CMAKE.write_text(s[:m.start()] + addition + s[m.end():])
    print("[dual-snapshot] added tartanair_stereo_dual_eval CMake target")


if __name__ == "__main__":
    patch_header()
    patch_source()
    patch_cmake()
    print("[dual-snapshot] done")
