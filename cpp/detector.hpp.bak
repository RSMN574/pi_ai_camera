#pragma once

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
        float confThreshold = 0.4f,
        float nmsThreshold = 0.45f
    );

    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;

    float confThreshold;
    float nmsThreshold;

    const int inputWidth = 640;
    const int inputHeight = 640;
};
