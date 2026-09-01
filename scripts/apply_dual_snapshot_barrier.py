#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "gaussian_mapper.h"
SOURCE = ROOT / "src" / "gaussian_mapper.cpp"
CMAKE = ROOT / "CMakeLists.txt"

HEADER_MARK = "waitForEvaluationSnapshotReady"
SOURCE_MARK = "[dual snapshot] online boundary ready"
TARGET_MARK = "tartanair_stereo_dual_eval"


def patch_header():
    s = HEADER.read_text()
    if HEADER_MARK in s:
        print("[dual-snapshot] gaussian_mapper.h already patched")
        return

    old = """    bool isStopped();\n    void signalStop(const bool going_to_stop = true);\n\n    int getIteration();\n"""
    new = """    bool isStopped();\n    void signalStop(const bool going_to_stop = true);\n\n    // Evaluation-only barrier used by the TartanAir dual-snapshot benchmark.\n    // It pauses the mapper exactly once after SLAM has shut down and final\n    // SLAM poses have been synchronized, before post-stream 30K training.\n    void enableEvaluationSnapshotBarrier(const bool enabled = true);\n    bool waitForEvaluationSnapshotReady();\n    void releaseEvaluationSnapshot();\n    int getEvaluationSnapshotIteration();\n\n    int getIteration();\n"""
    if old not in s:
        raise SystemExit("ERROR: gaussian_mapper.h public insertion point not found")
    s = s.replace(old, new, 1)

    old = """    std::mutex mutex_status_;\n    std::mutex mutex_settings_;\n    std::mutex mutex_render_; \n    \n};\n"""
    new = """    std::mutex mutex_status_;\n    std::mutex mutex_settings_;\n    std::mutex mutex_render_;\n\n    // Evaluation-only dual-snapshot synchronization. Disabled by default,\n    // therefore normal SEGS-SLAM runs are behavior-identical to upstream.\n    bool evaluation_snapshot_barrier_enabled_ = false;\n    bool evaluation_snapshot_ready_ = false;\n    bool evaluation_snapshot_released_ = false;\n    int evaluation_snapshot_iteration_ = -1;\n    std::mutex mutex_evaluation_snapshot_;\n    std::condition_variable cv_evaluation_snapshot_;\n    \n};\n"""
    if old not in s:
        raise SystemExit("ERROR: gaussian_mapper.h protected insertion point not found")
    s = s.replace(old, new, 1)
    HEADER.write_text(s)
    print("[dual-snapshot] patched include/gaussian_mapper.h")


def patch_source():
    s = SOURCE.read_text()
    if SOURCE_MARK not in s:
        old = """void GaussianMapper::signalStop(const bool going_to_stop)\n{\n    std::unique_lock<std::mutex> lock_status(this->mutex_status_);\n    this->stopped_ = going_to_stop;\n}\n"""
        new = old + r'''
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
        if old not in s:
            raise SystemExit("ERROR: gaussian_mapper.cpp signalStop insertion point not found")
        s = s.replace(old, new, 1)

        old = """            if(light_mode)\n                break;\n"""
        new = r'''            // Evaluation-only pause point. At this point the input stream has ended,
            // ORB-SLAM3 has finished, and final SLAM poses have been synchronized
            // into Gaussian keyframes. No post-stream training iteration has been
            // started after this boundary.
            {
                std::unique_lock<std::mutex> snapshot_lock(mutex_evaluation_snapshot_);
                if (evaluation_snapshot_barrier_enabled_) {
                    evaluation_snapshot_iteration_ = getIteration();
                    evaluation_snapshot_ready_ = true;
                    std::cout << "\n[dual snapshot] online boundary ready at iteration "
                              << evaluation_snapshot_iteration_ << std::endl;
                    cv_evaluation_snapshot_.notify_all();
                    cv_evaluation_snapshot_.wait(snapshot_lock, [this]() {
                        return evaluation_snapshot_released_;
                    });
                    std::cout << "[dual snapshot] released; continuing original post-stream optimization"
                              << std::endl;
                }
            }

            if(light_mode)
                break;
'''
        if old not in s:
            raise SystemExit("ERROR: gaussian_mapper.cpp online boundary insertion point not found")
        s = s.replace(old, new, 1)
        SOURCE.write_text(s)
        print("[dual-snapshot] patched src/gaussian_mapper.cpp")
    else:
        print("[dual-snapshot] gaussian_mapper.cpp already patched")


def patch_cmake():
    s = CMAKE.read_text()
    if TARGET_MARK in s:
        print("[dual-snapshot] CMake target already present")
        return
    old = """target_link_libraries(tartanair_stereo_eval\n    gaussian_viewer\n    gaussian_mapper\n    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)\n"""
    new = old + """\n# TartanAir one-run online + full30K evaluator (evaluation-only barrier)\nadd_executable(tartanair_stereo_dual_eval examples/tartanair_stereo_dual_eval.cpp)\ntarget_link_libraries(tartanair_stereo_dual_eval\n    gaussian_viewer\n    gaussian_mapper\n    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)\n"""
    if old not in s:
        raise SystemExit("ERROR: CMake tartanair_stereo_eval target not found")
    CMAKE.write_text(s.replace(old, new, 1))
    print("[dual-snapshot] added tartanair_stereo_dual_eval CMake target")


if __name__ == "__main__":
    patch_header()
    patch_source()
    patch_cmake()
    print("[dual-snapshot] done")
