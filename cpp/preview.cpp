#include <libcamera/libcamera.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>

#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include "object_tracker.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace libcamera;

namespace {

constexpr const char* kWindowName = "Raspberry Pi Camera";

std::mutex gClickMutex;
cv::Point gClickedPoint(-1, -1);
bool gClickPending = false;

void onMouse(int event, int x, int y, int, void*)
{
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    std::lock_guard<std::mutex> lock(gClickMutex);
    gClickedPoint = cv::Point(x, y);
    gClickPending = true;
}

bool takeClick(cv::Point& point)
{
    std::lock_guard<std::mutex> lock(gClickMutex);

    if (!gClickPending) {
        return false;
    }

    point = gClickedPoint;
    gClickPending = false;
    return true;
}

std::string classLabel(const Detection& detection,
                       const std::vector<std::string>& names)
{
    std::string label = "class " + std::to_string(detection.classId);

    if (detection.classId >= 0 &&
        detection.classId < static_cast<int>(names.size())) {
        label = names[static_cast<size_t>(detection.classId)];
    }

    label += " " + std::to_string(
        static_cast<int>(detection.confidence * 100.0f)) + "%";

    return label;
}

} // namespace

class CameraPreview
{
public:
    CameraPreview() = default;

    int run()
    {
        std::cerr << "[INFO] Loading YOLOv8n model..." << std::endl;
        YOLODetector detector("models/yolov8n.onnx", 0.40f, 0.45f);
        std::cerr << "[INFO] YOLOv8n model loaded." << std::endl;

        int ret = cameraManager_.start();
        if (ret) {
            std::cerr << "CameraManager start failed: " << ret << std::endl;
            return ret;
        }

        if (cameraManager_.cameras().empty()) {
            std::cerr << "No camera found" << std::endl;
            cameraManager_.stop();
            return -1;
        }

        camera_ = cameraManager_.cameras()[0];
        std::cout << "Using camera: " << camera_->id() << std::endl;

        ret = camera_->acquire();
        if (ret) {
            std::cerr << "Camera acquire failed: " << ret << std::endl;
            cameraManager_.stop();
            return ret;
        }

        config_ = camera_->generateConfiguration({ StreamRole::Viewfinder });
        if (!config_) {
            std::cerr << "Failed to generate camera configuration" << std::endl;
            cleanup();
            return -1;
        }

        streamConfig_ = &config_->at(0);
        streamConfig_->size = Size(640, 480);
        streamConfig_->pixelFormat = formats::RGB888;

        const CameraConfiguration::Status validation = config_->validate();
        if (validation == CameraConfiguration::Invalid) {
            std::cerr << "Invalid camera configuration" << std::endl;
            cleanup();
            return -1;
        }

        ret = camera_->configure(config_.get());
        if (ret) {
            std::cerr << "Camera configure failed: " << ret << std::endl;
            cleanup();
            return ret;
        }

        stream_ = streamConfig_->stream();

        std::cout << "Resolution: "
                  << streamConfig_->size.width << "x"
                  << streamConfig_->size.height << std::endl;
        std::cout << "Format: " << streamConfig_->pixelFormat.toString()
                  << std::endl;

        allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
        ret = allocator_->allocate(stream_);
        if (ret < 0) {
            std::cerr << "FrameBuffer allocation failed: " << ret << std::endl;
            cleanup();
            return ret;
        }

        const auto& buffers = allocator_->buffers(stream_);
        std::cout << "Buffers: " << buffers.size() << std::endl;

        mappedBuffers_.reserve(buffers.size());

        for (const auto& buffer : buffers) {
            // Add the entry first so cleanup can unmap already-mapped planes
            // if a later mmap call fails.
            mappedBuffers_.emplace_back();
            auto& mapped = mappedBuffers_.back();

            for (const auto& plane : buffer->planes()) {
                void* data = mmap(nullptr,
                                  plane.length,
                                  PROT_READ,
                                  MAP_SHARED,
                                  plane.fd.get(),
                                  plane.offset);

                if (data == MAP_FAILED) {
                    std::cerr << "mmap failed" << std::endl;
                    cleanup();
                    return -1;
                }

                mapped.push_back({ data, plane.length });
            }
        }

        camera_->requestCompleted.connect(this, &CameraPreview::requestComplete);

        for (const auto& buffer : buffers) {
            std::unique_ptr<Request> request = camera_->createRequest();
            if (!request) {
                std::cerr << "Failed to create request" << std::endl;
                cleanup();
                return -1;
            }

            ret = request->addBuffer(stream_, buffer.get());
            if (ret < 0) {
                std::cerr << "Failed to add buffer: " << ret << std::endl;
                cleanup();
                return ret;
            }

            requests_.push_back(std::move(request));
        }

        ret = camera_->start();
        if (ret) {
            std::cerr << "Camera start failed: " << ret << std::endl;
            cleanup();
            return ret;
        }
        cameraStarted_ = true;
        running_.store(true);

        for (auto& request : requests_) {
            ret = camera_->queueRequest(request.get());
            if (ret < 0) {
                std::cerr << "Failed to queue request: " << ret << std::endl;
                running_.store(false);
                condition_.notify_all();
                camera_->stop();
                cameraStarted_ = false;
                cleanup();
                return ret;
            }
        }

        std::cout << "\n====================================\n"
                  << " Raspberry Pi Camera + YOLOv8n + KCF\n"
                  << " Left click: select a detected object\n"
                  << " R: reset / select another object\n"
                  << " Q: quit\n"
                  << "====================================\n";

        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowName, 640, 480);
        cv::setMouseCallback(kWindowName, onMouse);

        ObjectTracker tracker;
        std::vector<Detection> detections;

        int frameCount = 0;
        double displayFps = 0.0;
        auto lastFpsTime = std::chrono::steady_clock::now();

        int inferenceFrames = 0;
        double inferenceTotalMs = 0.0;
        auto inferenceStatsStart = std::chrono::steady_clock::now();

        int exitCode = 0;

        while (running_.load()) {
            cv::Mat frame;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this]() {
                    return frameReady_ || !running_.load();
                });

                if (!running_.load()) {
                    break;
                }

                frame = latestFrame_.clone();
                frameReady_ = false;
            }

            if (frame.empty()) {
                continue;
            }

            // YOLO runs only while waiting for a selection. KCF handles
            // frame-to-frame tracking after the user selects a detection.
            if (tracker.state() == TrackingState::WAITING) {
                const auto inferStart = std::chrono::steady_clock::now();

                try {
                    detections = detector.detect(frame);
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] YOLO inference failed: "
                              << e.what() << std::endl;
                    exitCode = 1;
                    running_.store(false);
                    break;
                }

                const auto inferEnd = std::chrono::steady_clock::now();
                const double inferMs =
                    std::chrono::duration<double, std::milli>(
                        inferEnd - inferStart).count();

                ++inferenceFrames;
                inferenceTotalMs += inferMs;

                const double inferElapsed =
                    std::chrono::duration<double>(
                        inferEnd - inferenceStatsStart).count();

                if (inferElapsed >= 1.0) {
                    std::cerr << "[PERF] YOLO FPS: "
                              << inferenceFrames / inferElapsed
                              << " | average inference: "
                              << inferenceTotalMs / inferenceFrames
                              << " ms" << std::endl;

                    inferenceFrames = 0;
                    inferenceTotalMs = 0.0;
                    inferenceStatsStart = inferEnd;
                }
            } else if (tracker.state() == TrackingState::TRACKING) {
                if (!tracker.update(frame)) {
                    std::cerr << "[INFO] Target lost. Last box retained; "
                                 "press R to select again."
                              << std::endl;
                }
            }

            cv::Mat display = frame.clone();

            if (tracker.state() == TrackingState::WAITING) {
                for (const auto& detection : detections) {
                    cv::rectangle(display, detection.box,
                                  cv::Scalar(0, 255, 0), 2);

                    cv::putText(display,
                                classLabel(detection, detector.classNames()),
                                cv::Point(detection.box.x,
                                          std::max(20, detection.box.y - 8)),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.5,
                                cv::Scalar(0, 255, 0),
                                1);
                }

                cv::putText(display, "Click a detection to track",
                            cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(0, 255, 255), 2);

                cv::Point click;
                if (takeClick(click)) {
                    bool selected = false;

                    // If boxes overlap, choose the smallest containing box,
                    // which is usually the more specific target.
                    const Detection* chosen = nullptr;
                    for (const auto& detection : detections) {
                        if (detection.box.contains(click) &&
                            (!chosen ||
                             detection.box.area() < chosen->box.area())) {
                            chosen = &detection;
                        }
                    }

                    if (chosen) {
                        selected = tracker.select(
                            frame, chosen->box, chosen->classId);

                        if (selected) {
                            std::cout << "[INFO] Tracking selected object: "
                                      << classLabel(*chosen,
                                                    detector.classNames())
                                      << std::endl;
                        } else {
                            std::cerr << "[WARN] Could not initialize KCF "
                                         "for this detection."
                                      << std::endl;
                        }
                    } else {
                        std::cout << "[INFO] Click was not inside a detection "
                                     "box; click a green box."
                                  << std::endl;
                    }
                }
            } else if (tracker.state() == TrackingState::TRACKING) {
                const cv::Rect& box = tracker.box();
                cv::rectangle(display, box, cv::Scalar(255, 180, 0), 2);

                std::string label = "TRACKING";
                const int classId = tracker.classId();
                const auto& names = detector.classNames();
                if (classId >= 0 && classId < static_cast<int>(names.size())) {
                    label += ": " + names[static_cast<size_t>(classId)];
                }

                cv::putText(display, label,
                            cv::Point(box.x, std::max(20, box.y - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 180, 0), 2);
            } else {
                const cv::Rect& lastBox = tracker.lastBox();
                if (lastBox.width > 0 && lastBox.height > 0) {
                    cv::rectangle(display, lastBox,
                                  cv::Scalar(0, 0, 255), 2);
                }

                cv::putText(display, "TARGET LOST - Press R to reset",
                            cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(0, 0, 255), 2);
            }

            ++frameCount;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - lastFpsTime).count();

            if (elapsed >= 1.0) {
                displayFps = frameCount / elapsed;
                frameCount = 0;
                lastFpsTime = now;
            }

            cv::putText(display,
                        "FPS: " + std::to_string(
                            static_cast<int>(displayFps)),
                        cv::Point(10, 56),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 0), 2);

            cv::imshow(kWindowName, display);

            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q' || key == 27) {
                running_.store(false);
                condition_.notify_all();
                break;
            }

            if (key == 'r' || key == 'R') {
                tracker.reset();
                detections.clear();
                {
                    std::lock_guard<std::mutex> lock(gClickMutex);
                    gClickPending = false;
                }
                std::cout << "[INFO] Tracker reset. Select an object again."
                          << std::endl;
            }
        }

        running_.store(false);
        condition_.notify_all();

        if (cameraStarted_ && camera_) {
            camera_->stop();
            cameraStarted_ = false;
        }

        cv::destroyAllWindows();
        cleanup();
        return exitCode;
    }

private:
    struct MappedPlane {
        void* data = nullptr;
        size_t length = 0;
    };

    void requestComplete(Request* request)
    {
        if (request->status() == Request::RequestCancelled) {
            return;
        }

        const auto buffers = request->buffers();
        const auto it = buffers.find(stream_);
        if (it == buffers.end()) {
            requeueRequest(request);
            return;
        }

        FrameBuffer* buffer = it->second;
        const auto& allocatedBuffers = allocator_->buffers(stream_);

        size_t index = allocatedBuffers.size();
        for (size_t i = 0; i < allocatedBuffers.size(); ++i) {
            if (allocatedBuffers[i].get() == buffer) {
                index = i;
                break;
            }
        }

        if (index >= mappedBuffers_.size() ||
            mappedBuffers_[index].empty()) {
            requeueRequest(request);
            return;
        }

        void* data = mappedBuffers_[index][0].data;
        cv::Mat bgr;

        try {
            cv::Mat rgb(480, 640, CV_8UC3,
                        data, streamConfig_->stride);
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        } catch (const cv::Exception& e) {
            std::cerr << "[WARN] Frame conversion failed: "
                      << e.what() << std::endl;
            requeueRequest(request);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            latestFrame_ = bgr.clone();
            frameReady_ = true;
        }
        condition_.notify_one();

        requeueRequest(request);
    }

    void requeueRequest(Request* request)
    {
        if (!running_.load() || !camera_) {
            return;
        }

        request->reuse(Request::ReuseBuffers);
        const int ret = camera_->queueRequest(request);

        if (ret < 0) {
            std::cerr << "[ERROR] Failed to requeue request: "
                      << ret << std::endl;
        }
    }

    void cleanup()
    {
        for (auto& buffer : mappedBuffers_) {
            for (auto& plane : buffer) {
                if (plane.data) {
                    munmap(plane.data, plane.length);
                    plane.data = nullptr;
                }
            }
        }
        mappedBuffers_.clear();

        allocator_.reset();
        requests_.clear();

        if (camera_) {
            camera_->release();
            camera_.reset();
        }

        cameraManager_.stop();
        streamConfig_ = nullptr;
        stream_ = nullptr;
    }

    CameraManager cameraManager_;
    std::shared_ptr<Camera> camera_;
    std::unique_ptr<CameraConfiguration> config_;
    StreamConfiguration* streamConfig_ = nullptr;
    Stream* stream_ = nullptr;
    std::unique_ptr<FrameBufferAllocator> allocator_;
    std::vector<std::unique_ptr<Request>> requests_;
    std::vector<std::vector<MappedPlane>> mappedBuffers_;

    cv::Mat latestFrame_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> running_{false};
    bool frameReady_ = false;
    bool cameraStarted_ = false;
};

int main()
{
    try {
        CameraPreview preview;
        return preview.run();
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Application error: " << e.what() << std::endl;
        return 1;
    }
}
