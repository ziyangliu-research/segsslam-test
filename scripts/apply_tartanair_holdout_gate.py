#!/usr/bin/env python3
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
p = repo / "ORB-SLAM3" / "src" / "Tracking.cc"
s = p.read_text()

marker = "SEGS_HOLDOUT_PERIOD"
if marker in s and "suppressMappingForHoldoutFrame" in s:
    print(f"Holdout keyframe gate already applied: {p}")
    sys.exit(0)

old = "#include <mutex>\n#include <chrono>\n"
new = "#include <mutex>\n#include <chrono>\n#include <cstdlib>\n"
if old not in s:
    raise SystemExit("ERROR: include insertion point not found in Tracking.cc")
s = s.replace(old, new, 1)

old = "namespace ORB_SLAM3\n{\n\n\nTracking::Tracking"
new = r'''namespace ORB_SLAM3
{

namespace
{
bool tartanAirHoldoutGateEnabled()
{
    return std::getenv("SEGS_HOLDOUT_PERIOD") != nullptr &&
           std::getenv("SEGS_HOLDOUT_OFFSET") != nullptr;
}

bool suppressMappingForHoldoutFrame(const unsigned long frame_id)
{
    const char* period_env = std::getenv("SEGS_HOLDOUT_PERIOD");
    const char* offset_env = std::getenv("SEGS_HOLDOUT_OFFSET");
    if (!period_env || !offset_env)
        return false;

    char* period_end = nullptr;
    char* offset_end = nullptr;
    const long period = std::strtol(period_env, &period_end, 10);
    const long offset = std::strtol(offset_env, &offset_end, 10);

    if (!period_end || *period_end != '\0' || !offset_end || *offset_end != '\0' ||
        period <= 0 || offset < 0 || offset >= period)
        return false;

    return (frame_id % static_cast<unsigned long>(period)) ==
           static_cast<unsigned long>(offset);
}
}


Tracking::Tracking'''
if old not in s:
    raise SystemExit("ERROR: namespace insertion point not found in Tracking.cc")
s = s.replace(old, new, 1)

old = "void Tracking::StereoInitialization()\n{\n    if(mCurrentFrame.N>500)"
new = '''void Tracking::StereoInitialization()
{
    // Evaluation-only holdout gate: TEST frames may track an existing map but
    // must never initialize a new map.
    if (suppressMappingForHoldoutFrame(mCurrentFrame.mnId))
        return;

    if(mCurrentFrame.N>500)'''
if old not in s:
    raise SystemExit("ERROR: StereoInitialization insertion point not found")
s = s.replace(old, new, 1)

old = "bool Tracking::NeedNewKeyFrame()\n{\n    if((mSensor == System::IMU_MONOCULAR"
new = '''bool Tracking::NeedNewKeyFrame()
{
    // Evaluation-only holdout gate. Normal pose estimation and TrackLocalMap()
    // have already run; suppress only map/keyframe insertion for TEST frames.
    if (suppressMappingForHoldoutFrame(mCurrentFrame.mnId))
        return false;

    if((mSensor == System::IMU_MONOCULAR'''
if old not in s:
    raise SystemExit("ERROR: NeedNewKeyFrame insertion point not found")
s = s.replace(old, new, 1)

old = '''void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}'''
new = '''void Tracking::InformOnlyTracking(const bool &flag)
{
    // The benchmark's old implementation toggled localization-only mode on
    // TEST frames. With the dedicated holdout gate enabled, keep the original
    // SLAM tracking path active and suppress only keyframe insertion.
    if (tartanAirHoldoutGateEnabled()) {
        mbOnlyTracking = false;
        return;
    }
    mbOnlyTracking = flag;
}'''
if old not in s:
    raise SystemExit("ERROR: InformOnlyTracking replacement point not found")
s = s.replace(old, new, 1)

p.write_text(s)
print(f"Applied TartanAir holdout keyframe gate: {p}")
print("Changed behavior only when SEGS_HOLDOUT_PERIOD and SEGS_HOLDOUT_OFFSET are set.")
