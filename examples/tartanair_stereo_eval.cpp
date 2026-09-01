#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
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
    double psnr = std::numeric_limits<double>::quiet_NaN();
    double ssim = std::numeric_limits<double>::quiet_NaN();
};

void loadImages(const fs::path& cam0,
                const fs::path& cam1,
                const fs::path& timestamp_file,
                std::vector<std::string>& left,
                std::vector<std::string>& right,
                std::vector<double>& timestamps)
{
    std::ifstream f(timestamp_file);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open timestamp file: " + timestamp_file.string());
    }

    std::string line;
    while (std::getline(f, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c) { return c == '\r' || c == '\n'; }), line.end());
        if (line.empty() || line[0] == '#')
            continue;

        const fs::path l = cam0 / (line + ".png");
        const fs::path r = cam1 / (line + ".png");
        left.push_back(l.string());
        right.push_back(r.string());
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

void writeRunStats(const fs::path& output_dir,
                   const std::string& sequence,
                   int n_images,
                   int tracked_count,
                   int maxmap_count,
                   double sum_track_sec,
                   double stream_wall_sec,
                   double pipeline_wall_sec,
                   int iteration,
                   long long anchors,
                   long long gaussians,
                   int split_period,
                   int split_offset,
                   bool realtime_pacing,
                   bool render_metrics)
{
    std::ofstream f(output_dir / "run_stats.json");
    f << std::fixed << std::setprecision(9);
    f << "{\n";
    f << "  \"sequence\": \"" << sequence << "\",\n";
    f << "  \"frames\": " << n_images << ",\n";
    f << "  \"tracked_frames\": " << tracked_count << ",\n";
    f << "  \"largest_map_frames\": " << maxmap_count << ",\n";
    f << "  \"maxmap_ratio\": " << (n_images > 0 ? static_cast<double>(maxmap_count) / n_images : 0.0) << ",\n";
    f << "  \"tracking_call_seconds\": " << sum_track_sec << ",\n";
    f << "  \"tracking_call_fps\": " << (sum_track_sec > 0.0 ? n_images / sum_track_sec : 0.0) << ",\n";
    f << "  \"stream_wall_seconds\": " << stream_wall_sec << ",\n";
    f << "  \"stream_wall_fps\": " << (stream_wall_sec > 0.0 ? n_images / stream_wall_sec : 0.0) << ",\n";
    f << "  \"pipeline_wall_seconds\": " << pipeline_wall_sec << ",\n";
    f << "  \"pipeline_fps\": " << (pipeline_wall_sec > 0.0 ? n_images / pipeline_wall_sec : 0.0) << ",\n";
    f << "  \"final_iteration\": " << iteration << ",\n";
    f << "  \"anchors\": " << anchors << ",\n";
    f << "  \"gaussians\": " << gaussians << ",\n";
    f << "  \"n_offsets\": " << (anchors > 0 ? gaussians / anchors : 0) << ",\n";
    f << "  \"split_period\": " << split_period << ",\n";
    f << "  \"split_offset\": " << split_offset << ",\n";
    f << "  \"realtime_pacing\": " << (realtime_pacing ? "true" : "false") << ",\n";
    f << "  \"render_metrics\": " << (render_metrics ? "true" : "false") << "\n";
    f << "}\n";
}

void writeMetricSummary(const fs::path& output_dir,
                        int train_count,
                        double train_psnr,
                        double train_ssim,
                        int test_count,
                        double test_psnr,
                        double test_ssim)
{
    std::ofstream f(output_dir / "render_summary.json");
    f << std::fixed << std::setprecision(9);
    f << "{\n";
    f << "  \"train_count\": " << train_count << ",\n";
    f << "  \"train_psnr\": " << train_psnr << ",\n";
    f << "  \"train_ssim\": " << train_ssim << ",\n";
    f << "  \"test_count\": " << test_count << ",\n";
    f << "  \"test_psnr\": " << test_psnr << ",\n";
    f << "  \"test_ssim\": " << test_ssim << "\n";
    f << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 12) {
        std::cerr
            << "Usage: " << argv[0]
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

    if (split_period <= 0 || split_offset < 0 || split_offset >= split_period) {
        std::cerr << "Invalid split: period=" << split_period << " offset=" << split_offset << std::endl;
        return 2;
    }
    if (pace_mode != "realtime" && pace_mode != "compute") {
        std::cerr << "pace mode must be realtime or compute" << std::endl;
        return 2;
    }
    if (render_mode != "render" && render_mode != "no_render") {
        std::cerr << "render mode must be render or no_render" << std::endl;
        return 2;
    }

    const bool realtime_pacing = pace_mode == "realtime";
    const bool render_metrics = render_mode == "render";
    fs::create_directories(output_dir);

    std::vector<std::string> left_images, right_images;
    std::vector<double> timestamps;
    loadImages(sequence_root / "mav0/cam0/data",
               sequence_root / "mav0/cam1/data",
               timestamp_file,
               left_images,
               right_images,
               timestamps);

    if (left_images.empty() || left_images.size() != right_images.size() || left_images.size() != timestamps.size()) {
        std::cerr << "Invalid image/timestamp lists" << std::endl;
        return 3;
    }

    const int n_images = static_cast<int>(left_images.size());
    std::vector<FrameRecord> records(n_images);
    std::vector<double> tracking_times(n_images, 0.0);

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << (device_type == torch::kCUDA ? "CUDA available! Training on GPU." : "Training on CPU.") << std::endl;

    auto pSLAM = std::make_shared<ORB_SLAM3::System>(
        vocab.string(), orb_cfg.string(), ORB_SLAM3::System::STEREO);
    const float image_scale = pSLAM->GetImageScale();

    auto pGausMapper = std::make_shared<GaussianMapper>(
        pSLAM, gaus_cfg, output_dir, 0, device_type);
    std::thread training_thread(&GaussianMapper::run, pGausMapper.get());

    std::cout << "\n============================================================\n";
    std::cout << "SEGS-SLAM TartanAir holdout evaluation\n";
    std::cout << "Sequence      : " << sequence << "\n";
    std::cout << "Frames        : " << n_images << "\n";
    std::cout << "Split         : every " << split_period << " frames, offset " << split_offset << " is TEST\n";
    std::cout << "Pacing        : " << pace_mode << "\n";
    std::cout << "Render metrics: " << (render_metrics ? "yes" : "no") << "\n";
    std::cout << "============================================================\n\n";

    double sum_track_sec = 0.0;
    const auto pipeline_start = std::chrono::steady_clock::now();
    const auto stream_start = pipeline_start;

    for (int i = 0; i < n_images; ++i) {
        cv::Mat left = cv::imread(left_images[i], cv::IMREAD_UNCHANGED);
        cv::Mat right = cv::imread(right_images[i], cv::IMREAD_UNCHANGED);
        if (left.empty() || right.empty()) {
            std::cerr << "Failed to load stereo pair at frame " << i << std::endl;
            pSLAM->getTracker()->InformOnlyTracking(false);
            pSLAM->Shutdown();
            training_thread.join();
            return 4;
        }

        if (image_scale != 1.f) {
            const cv::Size size(static_cast<int>(left.cols * image_scale), static_cast<int>(left.rows * image_scale));
            cv::resize(left, left, size);
            cv::resize(right, right, size);
        }

        const bool is_test = (i % split_period) == split_offset;
        // IMPORTANT: use ORB-SLAM3's existing pose-only tracking flag for holdout frames.
        // No keyframe is inserted from a test frame, while the core tracking/mapping code is untouched.
        pSLAM->getTracker()->InformOnlyTracking(is_test);

        const auto t1 = std::chrono::steady_clock::now();
        pSLAM->TrackStereo(left, right, timestamps[i], {}, left_images[i]);
        const auto t2 = std::chrono::steady_clock::now();
        const double track_sec = std::chrono::duration<double>(t2 - t1).count();
        tracking_times[i] = track_sec;
        sum_track_sec += track_sec;

        const int state = pSLAM->GetTrackingState();
        records[i].tracked = (state == ORB_SLAM3::Tracking::OK || state == ORB_SLAM3::Tracking::OK_KLT);

        if ((i + 1) % 100 == 0 || i + 1 == n_images) {
            std::cout << "[stream] " << (i + 1) << "/" << n_images
                      << "  split=" << (is_test ? "test" : "train")
                      << "  state=" << state << std::endl;
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

    pSLAM->getTracker()->InformOnlyTracking(false);
    const auto stream_end = std::chrono::steady_clock::now();
    const double stream_wall_sec = std::chrono::duration<double>(stream_end - stream_start).count();

    // Preserve the original SEGS-SLAM hand-off: save the current SLAM trajectory,
    // expose it to GaussianMapper, then shut SLAM down so the mapper can perform
    // its configured tail optimization (light mode or full 30K).
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM_bf.txt").string());
    example_utils::LoadTrajectory((output_dir / "CameraTrajectory_TUM_bf.txt").string(), pGausMapper->pose_);
    pGausMapper->poseSaved = true;

    pSLAM->Shutdown();
    training_thread.join();
    const auto pipeline_end = std::chrono::steady_clock::now();
    const double pipeline_wall_sec = std::chrono::duration<double>(pipeline_end - pipeline_start).count();

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

    int train_count = 0, test_count = 0;
    double train_psnr_sum = 0.0, train_ssim_sum = 0.0;
    double test_psnr_sum = 0.0, test_ssim_sum = 0.0;

    if (render_metrics) {
        torch::NoGradGuard no_grad;
        pGausMapper->gaussians_->eval();
        for (int i = 0; i < n_images; ++i) {
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

            cv::Mat rendered_rgb = pGausMapper->renderFromPose(
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

            records[i].psnr = computePSNR(rendered_rgb, gt_rgb);
            records[i].ssim = computeSSIM(rendered_rgb, gt_rgb);

            const bool is_test = (i % split_period) == split_offset;
            if (is_test) {
                ++test_count;
                test_psnr_sum += records[i].psnr;
                test_ssim_sum += records[i].ssim;
            } else {
                ++train_count;
                train_psnr_sum += records[i].psnr;
                train_ssim_sum += records[i].ssim;
            }

            if ((i + 1) % 100 == 0 || i + 1 == n_images) {
                std::cout << "[render] frame " << i
                          << "  " << (is_test ? "test" : "train")
                          << "  PSNR=" << records[i].psnr
                          << "  SSIM=" << records[i].ssim << std::endl;
            }
        }
    }

    const double train_psnr = train_count ? train_psnr_sum / train_count : 0.0;
    const double train_ssim = train_count ? train_ssim_sum / train_count : 0.0;
    const double test_psnr = test_count ? test_psnr_sum / test_count : 0.0;
    const double test_ssim = test_count ? test_ssim_sum / test_count : 0.0;

    std::ofstream pose_csv(output_dir / "per_frame_eval.csv");
    pose_csv << "frame,split,tracked,largest_map,tx,ty,tz,qx,qy,qz,qw,psnr,ssim\n";
    pose_csv << std::fixed << std::setprecision(9);
    for (int i = 0; i < n_images; ++i) {
        const bool is_test = (i % split_period) == split_offset;
        pose_csv << i << ',' << (is_test ? "test" : "train") << ','
                 << (records[i].tracked ? 1 : 0) << ','
                 << (records[i].largest_map ? 1 : 0) << ',';
        if (records[i].pose_valid) {
            const Sophus::SE3f Twc = records[i].Tcw.inverse();
            const auto t = Twc.translation();
            const auto q = Twc.unit_quaternion();
            pose_csv << t.x() << ',' << t.y() << ',' << t.z() << ','
                     << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w();
        } else {
            pose_csv << "nan,nan,nan,nan,nan,nan,nan";
        }
        pose_csv << ',' << records[i].psnr << ',' << records[i].ssim << '\n';
    }

    const long long anchors = pGausMapper->gaussians_->get_anchor().defined()
        ? pGausMapper->gaussians_->get_anchor().size(0) : 0;
    const long long n_offsets = pGausMapper->gaussians_->n_offsets;
    const long long gaussian_count = anchors * n_offsets;

    writeRunStats(output_dir, sequence, n_images, tracked_count, maxmap_count,
                  sum_track_sec, stream_wall_sec, pipeline_wall_sec,
                  pGausMapper->getIteration(), anchors, gaussian_count,
                  split_period, split_offset, realtime_pacing, render_metrics);
    writeMetricSummary(output_dir, train_count, train_psnr, train_ssim,
                       test_count, test_psnr, test_ssim);

    std::cout << "\n================ Result ================\n";
    std::cout << "MaxMap        : " << maxmap_count << '/' << n_images
              << " (" << std::setprecision(2) << std::fixed
              << (100.0 * maxmap_count / n_images) << "%)\n";
    if (render_metrics) {
        std::cout << std::setprecision(6);
        std::cout << "Train PSNR/SSIM: " << train_psnr << " / " << train_ssim << '\n';
        std::cout << "Test  PSNR/SSIM: " << test_psnr << " / " << test_ssim << '\n';
    }
    std::cout << "Anchors       : " << anchors << '\n';
    std::cout << "Gaussians     : " << gaussian_count << " (= anchors x " << n_offsets << ")\n";
    std::cout << "Stream FPS    : " << (stream_wall_sec > 0 ? n_images / stream_wall_sec : 0.0) << '\n';
    std::cout << "Pipeline FPS  : " << (pipeline_wall_sec > 0 ? n_images / pipeline_wall_sec : 0.0) << '\n';
    std::cout << "========================================\n";

    return 0;
}
