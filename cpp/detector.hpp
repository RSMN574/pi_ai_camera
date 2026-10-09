#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

class YOLODetector {
public:
    YOLODetector(
        const std::string& modelPath,
        float confThreshold = 0.40f,
        float nmsThreshold = 0.45f
    );

    std::vector<Detection> detect(const cv::Mat& frame);
    const std::vector<std::string>& classNames() const;

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memoryInfo_;

    std::string inputName_;
    std::string outputName_;

    float confThreshold_;
    float nmsThreshold_;

    static constexpr int inputWidth_ = 640;
    static constexpr int inputHeight_ = 640;

    std::vector<std::string> classNames_;
};
