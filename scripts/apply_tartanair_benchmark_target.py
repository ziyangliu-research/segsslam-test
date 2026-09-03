#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cmake = root / "CMakeLists.txt"
s = cmake.read_text()
mark = "add_executable(tartanair_stereo_benchmark"
if mark in s:
    print("[benchmark] CMake target already present")
    raise SystemExit(0)

block = r'''

# Final TartanAir benchmark: one tracking run, online snapshot, same mapper -> full30K.
add_executable(tartanair_stereo_benchmark examples/tartanair_stereo_benchmark.cpp)
target_link_libraries(tartanair_stereo_benchmark
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)
'''
cmake.write_text(s.rstrip() + block + "\n")
print("[benchmark] added tartanair_stereo_benchmark CMake target")
