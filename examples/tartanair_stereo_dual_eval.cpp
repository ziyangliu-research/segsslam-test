#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/Tracking.h"
#include "ORB-SLAM3/include/KeyFrame.h"
#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/Map.h"

#include "include/gaussian_mapper.h"
#include "example_utils.h"

namespace fs = std::filesystem;

namespace {

struct FrameRecord {
    bool tracked = false;
    bool largest_map = false;
    bool pose_valid = false;
    Sophus::SE3f Tcw = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
};

struct SnapshotMetrics {
    int train_count = 0;
    int test_count = 0;
    double train_psnr = 0.0;
    double train_ssim = 0.0;
    double test_psnr = 0.0;
    double test_ssim = 0.0;
    long long anchors = 0;
    long long gaussians = 0;
    int iteration = 0;
};

void loadImages(const fs::path& cam0,
                const fs::path& cam1,
                const fs::path& timestamp_file,
                std::vector<std::string>& left,
                std::vector<std::string>& right,
                std::vector<double>& timestamps)
{
    std::ifstream f(timestamp_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open timestamp file: " + timestamp_file.string());

    std::string line;
    while (std::getline(f, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c) {
            return c == '\r' || c == '\n';
        }), line.end());
        if (line.empty() || line[0] == '#')
            continue;

        left.push_back((cam0 / (line + ".png")).string());
        right.push_back((cam1 / (line + ".png")).string());
        timestamps.push_back(std::stod(line) / 1e9);
    }
}

double computePSNR(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff = diff.mul(diff);
    const cv::Scalar s = cv::sum(diff);
    const double sse = s[0] + s[1] + s[2];
    const double denom = static_cast<double>(a.total()) * a.channels();
    const double mse = sse / denom;
    if (mse <= 1e-12)
        return 120.0;
    return 10.0 * std::log10(1.0 / mse);
}

double computeSSIMSingle(const cv::Mat& a, const cv::Mat& b)
{
    constexpr double C1 = 0.01 * 0.01;
    constexpr double C2 = 0.03 * 0.03;
    cv::Mat g = cv::getGaussianKernel(11, 1.5, CV_32F);
    cv::Mat kernel = g * g.t();

    cv::Mat mu1, mu2;
    cv::filter2D(a, mu1, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(b, mu2, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat a_sq = a.mul(a);
    cv::Mat b_sq = b.mul(b);
    cv::Mat ab = a.mul(b);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::filter2D(a_sq, sigma1_sq, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(b_sq, sigma2_sq, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(ab, sigma12, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;
    sigma12 -= mu1_mu2;

    cv::Mat num1 = 2.0 * mu1_mu2;
    cv::add(num1, cv::Scalar::all(C1), num1);
    cv::Mat num2 = 2.0 * sigma12;
    cv::add(num2, cv::Scalar::all(C2), num2);
    cv::Mat numerator = num1.mul(num2);

    cv::Mat den1 = mu1_sq + mu2_sq;
    cv::add(den1, cv::Scalar::all(C1), den1);
    cv::Mat den2 = sigma1_sq + sigma2_sq;
    cv::add(den2, cv::Scalar::all(C2), den2);
    cv::Mat denominator = den1.mul(den2);

    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);
    return cv::mean(ssim_map)[0];
}

double computeSSIM(const cv::Mat& a, const cv::Mat& b)
{
    std::vector<cv::Mat> ac, bc;
    cv::split(a, ac);
    cv::split(b, bc);
    double total = 0.0;
    for (int c = 0; c < 3; ++c)
        total += computeSSIMSingle(ac[c], bc[c]);
    return total / 3.0;
}

ORB_SLAM3::Map* largestMap(ORB_SLAM3::Atlas* atlas)
{
    ORB_SLAM3::Map* best = nullptr;
    std::size_t best_kfs = 0;
    for (auto* map : atlas->GetAllMaps()) {
        if (!map)
            continue;
        const auto n = map->GetAllKeyFrames().size();
        if (n > best_kfs) {
            best_kfs = n;
            best = map;
        }
    }
    return best;
}

bool reconstructFinalTcw(ORB_SLAM3::KeyFrame* ref,
                         const Sophus::SE3f& relative,
                         ORB_SLAM3::Map* expected_map,
                         Sophus::SE3f& Tcw)
{
    if (!ref)
        return false;

    Sophus::SE3f Trw(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
    ORB_SLAM3::KeyFrame* kf = ref;
    while (kf && kf->isBad()) {
        Trw = Trw * kf->mTcp;
        kf = kf->GetParent();
    }
    if (!kf || (expected_map && kf->GetMap() != expected_map))
        return false;

    Trw = Trw * kf->GetPose();
    Tcw = relative * Trw;
    return true;
}

void writePerFrameCsv(const fs::path& path,
                      const std::vector<FrameRecord>& records,
                      const std::vector<double>& psnr,
                      const std::vector<double>& ssim,
                      int split_period,
                      int split_offset)
{
    std::ofstream f(path);
    f << "frame,split,tracked,largest_map,tx,ty,tz,qx,qy,qz,qw,psnr,ssim\n";
    f << std::fixed << std::setprecision(9);
    for (std::size_t i = 0; i < records.size(); ++i) {
        const bool is_test = (static_cast<int>(i) % split_period) == split_offset;
        f << i << ',' << (is_test ? "test" : "train") << ','
          << (records[i].tracked ? 1 : 0) << ','
          << (records[i].largest_map ? 1 : 0) << ',';
        if (records[i].pose_valid) {
            const Sophus::SE3f Twc = records[i].Tcw.inverse();
            const auto t = Twc.translation();
            const auto q = Twc.unit_quaternion();
            f << t.x() << ',' << t.y() << ',' << t.z() << ','
              << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w();
        } else {
            f << "nan,nan,nan,nan,nan,nan,nan";
        }
        f << ',' << psnr[i] << ',' << ssim[i] << '\n';
    }
}

void writeSnapshotJson(const fs::path& dir,
                       const std::string& sequence,
                       int n_images,
                       int tracked_count,
                       int maxmap_count,
                       int split_period,
                       int split_offset,
                       const SnapshotMetrics& m)
{
    std::ofstream f(dir / "run_stats.json");
    f << std::fixed << std::setprecision(9);
    f << "{\n";
    f << "  \"sequence\": \"" << sequence << "\",\n";
    f << "  \"frames\": " << n_images << ",\n";
    f << "  \"tracked_frames\": " << tracked_count << ",\n";
    f << "  \"largest_map_frames\": " << maxmap_count << ",\n";
    f << "  \"maxmap_ratio\": " << (n_images ? static_cast<double>(maxmap_count) / n_images : 0.0) << ",\n";
    f << "  \"final_iteration\": " << m.iteration << ",\n";
    f << "  \"anchors\": " << m.anchors << ",\n";
    f << "  \"gaussians\": " << m.gaussians << ",\n";
    f << "  \"n_offsets\": " << (m.anchors ? m.gaussians / m.anchors : 0) << ",\n";
    f << "  \"split_period\": " << split_period << ",\n";
    f << "  \"split_offset\": " << split_offset << "\n";
    f << "}\n";

    std::ofstream r(dir / "render_summary.json");
    r << std::fixed << std::setprecision(9);
    r << "{\n";
    r << "  \"train_count\": " << m.train_count << ",\n";
    r << "  \"train_psnr\": " << m.train_psnr << ",\n";
    r << "  \"train_ssim\": " << m.train_ssim << ",\n";
    r << "  \"test_count\": " << m.test_count << ",\n";
    r << "  \"test_psnr\": " << m.test_psnr << ",\n";
    r << "  \"test_ssim\": " << m.test_ssim << "\n";
    r << "}\n";
}

SnapshotMetrics evaluateSnapshot(const fs::path& snapshot_dir,
                                 const std::string& label,
                                 GaussianMapper& mapper,
                                 const std::vector<FrameRecord>& records,
                                 const std::vector<std::string>& left_images,
                                 float image_scale,
                                 int split_period,
                                 int split_offset,
                                 bool render_metrics,
                                 int tracked_count,
                                 int maxmap_count,
                                 const std::string& sequence)
{
    fs::create_directories(snapshot_dir);
    SnapshotMetrics m;
    m.iteration = mapper.getIteration();
    m.anchors = mapper.gaussians_->get_anchor().defined()
        ? mapper.gaussians_->get_anchor().size(0) : 0;
    const long long n_offsets = mapper.gaussians_->n_offsets;
    m.gaussians = m.anchors * n_offsets;

    std::vector<double> psnr(records.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> ssim(records.size(), std::numeric_limits<double>::quiet_NaN());
    double train_psnr_sum = 0.0, train_ssim_sum = 0.0;
    double test_psnr_sum = 0.0, test_ssim_sum = 0.0;

    if (render_metrics) {
        torch::NoGradGuard no_grad;
        mapper.gaussians_->eval();
        for (std::size_t i = 0; i < records.size(); ++i) {
            if (!records[i].largest_map || !records[i].pose_valid)
                continue;

            cv::Mat gt_bgr = cv::imread(left_images[i], cv::IMREAD_COLOR);
            if (gt_bgr.empty())
                continue;
            if (image_scale != 1.f) {
                cv::resize(gt_bgr, gt_bgr,
                           cv::Size(static_cast<int>(gt_bgr.cols * image_scale),
                                    static_cast<int>(gt_bgr.rows * image_scale)));
            }

            cv::Mat rendered_rgb = mapper.renderFromPose(
                records[i].Tcw, gt_bgr.cols, gt_bgr.rows, true);
            if (rendered_rgb.empty())
                continue;

            cv::Mat gt_rgb;
            cv::cvtColor(gt_bgr, gt_rgb, cv::COLOR_BGR2RGB);
            gt_rgb.convertTo(gt_rgb, CV_32FC3, 1.0 / 255.0);
            if (rendered_rgb.type() != CV_32FC3)
                rendered_rgb.convertTo(rendered_rgb, CV_32FC3);
            if (rendered_rgb.size() != gt_rgb.size())
                cv::resize(rendered_rgb, rendered_rgb, gt_rgb.size(), 0.0, 0.0, cv::INTER_LINEAR);
            cv::min(rendered_rgb, 1.0, rendered_rgb);
            cv::max(rendered_rgb, 0.0, rendered_rgb);

            psnr[i] = computePSNR(rendered_rgb, gt_rgb);
            ssim[i] = computeSSIM(rendered_rgb, gt_rgb);
            const bool is_test = (static_cast<int>(i) % split_period) == split_offset;
            if (is_test) {
                ++m.test_count;
                test_psnr_sum += psnr[i];
                test_ssim_sum += ssim[i];
            } else {
                ++m.train_count;
                train_psnr_sum += psnr[i];
                train_ssim_sum += ssim[i];
            }

            if ((i + 1) % 100 == 0 || i + 1 == records.size()) {
                std::cout << "[" << label << " render] " << (i + 1) << '/' << records.size()
                          << "  " << (is_test ? "test" : "train")
                          << "  PSNR=" << psnr[i] << " SSIM=" << ssim[i] << std::endl;
            }
        }
        mapper.gaussians_->train();
    }

    if (m.train_count) {
        m.train_psnr = train_psnr_sum / m.train_count;
        m.train_ssim = train_ssim_sum / m.train_count;
    }
    if (m.test_count) {
        m.test_psnr = test_psnr_sum / m.test_count;
        m.test_ssim = test_ssim_sum / m.test_count;
    }

    writePerFrameCsv(snapshot_dir / "per_frame_eval.csv", records, psnr, ssim,
                     split_period, split_offset);
    writeSnapshotJson(snapshot_dir, sequence, static_cast<int>(records.size()),
                      tracked_count, maxmap_count, split_period, split_offset, m);

    std::cout << "\n============ " << label << " snapshot ============\n";
    std::cout << "Iteration      : " << m.iteration << '\n';
    std::cout << "Anchors        : " << m.anchors << '\n';
    std::cout << "Gaussians      : " << m.gaussians << '\n';
    if (render_metrics) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Train PSNR/SSIM: " << m.train_psnr << " / " << m.train_ssim << '\n';
        std::cout << "Test  PSNR/SSIM: " << m.test_psnr << " / " << m.test_ssim << '\n';
    }
    std::cout << "==========================================\n";
    return m;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 12) {
        std::cerr << "Usage: " << argv[0]
                  << " VOCAB ORB_CFG GAUS_CFG SEQUENCE_ROOT TIMESTAMPS OUTPUT_DIR"
                  << " SEQUENCE_NAME SPLIT_PERIOD SPLIT_OFFSET realtime|compute render|no_render\n";
        return 1;
    }

    const fs::path vocab = argv[1];
    const fs::path orb_cfg = argv[2];
    const fs::path gaus_cfg = argv[3];
    const fs::path sequence_root = argv[4];
    const fs::path timestamp_file = argv[5];
    const fs::path output_dir = argv[6];
    const std::string sequence = argv[7];
    const int split_period = std::stoi(argv[8]);
    const int split_offset = std::stoi(argv[9]);
    const std::string pace_mode = argv[10];
    const std::string render_mode = argv[11];

    if (split_period <= 0 || split_offset < 0 || split_offset >= split_period)
        return 2;
    const bool realtime_pacing = pace_mode == "realtime";
    const bool render_metrics = render_mode == "render";
    fs::create_directories(output_dir);

    std::vector<std::string> left_images, right_images;
    std::vector<double> timestamps;
    loadImages(sequence_root / "mav0/cam0/data",
               sequence_root / "mav0/cam1/data",
               timestamp_file, left_images, right_images, timestamps);
    if (left_images.empty() || left_images.size() != right_images.size() || left_images.size() != timestamps.size())
        return 3;

    const int n_images = static_cast<int>(left_images.size());
    std::vector<FrameRecord> records(n_images);
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    auto pSLAM = std::make_shared<ORB_SLAM3::System>(
        vocab.string(), orb_cfg.string(), ORB_SLAM3::System::STEREO);
    const float image_scale = pSLAM->GetImageScale();

    auto pGausMapper = std::make_shared<GaussianMapper>(
        pSLAM, gaus_cfg, output_dir, 0, device_type);
    pGausMapper->enableEvaluationSnapshotBarrier(true);
    std::thread training_thread(&GaussianMapper::run, pGausMapper.get());

    std::cout << "\n============================================================\n";
    std::cout << "SEGS-SLAM TartanAir ONE-RUN online + full30K evaluation\n";
    std::cout << "Sequence : " << sequence << "\nFrames   : " << n_images << '\n';
    std::cout << "Split    : holdout_" << split_period << '_' << split_offset << '\n';
    std::cout << "============================================================\n";

    double sum_track_sec = 0.0;
    const auto total_start = std::chrono::steady_clock::now();
    const auto stream_start = total_start;

    for (int i = 0; i < n_images; ++i) {
        cv::Mat left = cv::imread(left_images[i], cv::IMREAD_UNCHANGED);
        cv::Mat right = cv::imread(right_images[i], cv::IMREAD_UNCHANGED);
        if (left.empty() || right.empty()) {
            std::cerr << "Failed to load stereo pair at frame " << i << std::endl;
            pGausMapper->releaseEvaluationSnapshot();
            pSLAM->Shutdown();
            training_thread.join();
            return 4;
        }
        if (image_scale != 1.f) {
            const cv::Size size(static_cast<int>(left.cols * image_scale),
                                static_cast<int>(left.rows * image_scale));
            cv::resize(left, left, size);
            cv::resize(right, right, size);
        }

        const auto t1 = std::chrono::steady_clock::now();
        // Holdout behavior is implemented by the evaluation-only KF gate in Tracking.cc.
        // Every frame follows the normal tracking path; TEST frames simply cannot create KFs.
        pSLAM->TrackStereo(left, right, timestamps[i], {}, left_images[i]);
        const auto t2 = std::chrono::steady_clock::now();
        const double track_sec = std::chrono::duration<double>(t2 - t1).count();
        sum_track_sec += track_sec;

        const int state = pSLAM->GetTrackingState();
        records[i].tracked = (state == ORB_SLAM3::Tracking::OK || state == ORB_SLAM3::Tracking::OK_KLT);

        if ((i + 1) % 100 == 0 || i + 1 == n_images) {
            const bool is_test = (i % split_period) == split_offset;
            std::cout << "[stream] " << (i + 1) << '/' << n_images
                      << " split=" << (is_test ? "test" : "train")
                      << " state=" << state << std::endl;
        }

        if (realtime_pacing) {
            double period = 0.0;
            if (i + 1 < n_images)
                period = timestamps[i + 1] - timestamps[i];
            else if (i > 0)
                period = timestamps[i] - timestamps[i - 1];
            if (period > track_sec)
                std::this_thread::sleep_for(std::chrono::duration<double>(period - track_sec));
        }
    }

    const auto stream_end = std::chrono::steady_clock::now();
    const double stream_wall_sec = std::chrono::duration<double>(stream_end - stream_start).count();

    // Keep the original SEGS-SLAM pose hand-off unchanged.
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM_bf.txt").string());
    example_utils::LoadTrajectory((output_dir / "CameraTrajectory_TUM_bf.txt").string(), pGausMapper->pose_);
    pGausMapper->poseSaved = true;

    // Shutdown waits for ORB-SLAM3 mapping/loop threads. The Gaussian mapper then
    // reaches the evaluation barrier after synchronizing final SLAM poses.
    pSLAM->Shutdown();
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());

    auto* big_map = largestMap(pSLAM->getAtlas());
    const auto refs = pSLAM->GetAllFrames();
    const auto rels = pSLAM->GetAllRelativeFramePoses();
    const std::size_t n_pose_records = std::min({refs.size(), rels.size(), static_cast<std::size_t>(n_images)});

    int tracked_count = 0;
    int maxmap_count = 0;
    for (int i = 0; i < n_images; ++i) {
        if (records[i].tracked)
            ++tracked_count;
        if (!records[i].tracked || static_cast<std::size_t>(i) >= n_pose_records)
            continue;
        Sophus::SE3f Tcw;
        if (reconstructFinalTcw(refs[i], rels[i], big_map, Tcw)) {
            records[i].largest_map = true;
            records[i].pose_valid = true;
            records[i].Tcw = Tcw;
            ++maxmap_count;
        }
    }

    if (!pGausMapper->waitForEvaluationSnapshotReady()) {
        std::cerr << "Gaussian mapper exited without reaching online snapshot barrier" << std::endl;
        training_thread.join();
        return 5;
    }

    std::cout << "\n[dual] ONLINE snapshot at iteration "
              << pGausMapper->getEvaluationSnapshotIteration() << std::endl;
    evaluateSnapshot(output_dir / "online", "ONLINE", *pGausMapper,
                     records, left_images, image_scale, split_period, split_offset,
                     render_metrics, tracked_count, maxmap_count, sequence);

    // Resume the exact original full-mode loop. No optimizer/densification state is reset.
    const auto post_start = std::chrono::steady_clock::now();
    pGausMapper->releaseEvaluationSnapshot();
    training_thread.join();
    const auto post_end = std::chrono::steady_clock::now();
    const double post_seconds = std::chrono::duration<double>(post_end - post_start).count();

    std::cout << "\n[dual] FULL30K snapshot at iteration " << pGausMapper->getIteration() << std::endl;
    evaluateSnapshot(output_dir / "full30k", "FULL30K", *pGausMapper,
                     records, left_images, image_scale, split_period, split_offset,
                     render_metrics, tracked_count, maxmap_count, sequence);

    const auto total_end = std::chrono::steady_clock::now();
    const double total_seconds = std::chrono::duration<double>(total_end - total_start).count();

    std::ofstream common(output_dir / "common_stats.json");
    common << std::fixed << std::setprecision(9);
    common << "{\n";
    common << "  \"sequence\": \"" << sequence << "\",\n";
    common << "  \"frames\": " << n_images << ",\n";
    common << "  \"tracked_frames\": " << tracked_count << ",\n";
    common << "  \"largest_map_frames\": " << maxmap_count << ",\n";
    common << "  \"maxmap_ratio\": " << (n_images ? static_cast<double>(maxmap_count) / n_images : 0.0) << ",\n";
    common << "  \"tracking_call_seconds\": " << sum_track_sec << ",\n";
    common << "  \"stream_wall_seconds\": " << stream_wall_sec << ",\n";
    common << "  \"post_optimization_seconds\": " << post_seconds << ",\n";
    common << "  \"total_seconds\": " << total_seconds << "\n";
    common << "}\n";

    std::cout << "\n================ COMMON =================\n";
    std::cout << "MaxMap       : " << maxmap_count << '/' << n_images
              << " (" << std::fixed << std::setprecision(2)
              << (100.0 * maxmap_count / n_images) << "%)\n";
    std::cout << "Stream wall  : " << stream_wall_sec << " s\n";
    std::cout << "Post 30K time: " << post_seconds << " s\n";
    std::cout << "=========================================\n";
    return 0;
}
