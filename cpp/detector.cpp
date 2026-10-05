#include "detector.hpp"

#include <algorithm>
#include <iostream>

YOLODetector::YOLODetector(
    const std::string& modelPath,
    float confThreshold,
    float nmsThreshold
)
    : confThreshold(confThreshold),
      nmsThreshold(nmsThreshold)
{
    net = cv::dnn::readNetFromONNX(modelPath);

    if (net.empty()) {
        throw std::runtime_error(
            "Failed to load YOLO ONNX model: " + modelPath
        );
    }

    // 純 OpenCV CPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "YOLO model loaded\n";
    std::cout << "Backend: OpenCV\n";
    std::cout << "Target: CPU\n";
}


std::vector<Detection> YOLODetector::detect(
    const cv::Mat& frame
)
{
    std::vector<Detection> detections;

    if (frame.empty()) {
        return detections;
    }

    // 建立 DNN input
    cv::Mat blob;

    cv::dnn::blobFromImage(
        frame,
        blob,
        1.0 / 255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(),
        true,
        false
    );

    net.setInput(blob);

    cv::Mat output = net.forward();

    /*
     * YOLOv5 ONNX 通常輸出：
     *
     * [x, y, width, height, confidence, classes...]
     */

    const int dimensions = output.size[2];
    const int rows = output.size[1];

    float* data = reinterpret_cast<float*>(output.data);

    float xFactor =
        static_cast<float>(frame.cols) / inputWidth;

    float yFactor =
        static_cast<float>(frame.rows) / inputHeight;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float* row = data + i * dimensions;

        float confidence = row[4];

        if (confidence < confThreshold) {
            continue;
        }

        cv::Mat scores(
            1,
            dimensions - 5,
            CV_32FC1,
            row + 5
        );

        cv::Point classIdPoint;
        double maxClassScore;

        cv::minMaxLoc(
            scores,
            nullptr,
            &maxClassScore,
            nullptr,
            &classIdPoint
        );

        double finalConfidence =
            confidence * maxClassScore;

        if (finalConfidence < confThreshold) {
            continue;
        }

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        int left =
            static_cast<int>(
                (cx - w / 2.0f) * xFactor
            );

        int top =
            static_cast<int>(
                (cy - h / 2.0f) * yFactor
            );

        int width =
            static_cast<int>(w * xFactor);

        int height =
            static_cast<int>(h * yFactor);

        cv::Rect box(
            left,
            top,
            width,
            height
        );

        box &= cv::Rect(
            0,
            0,
            frame.cols,
            frame.rows
        );

        classIds.push_back(
            classIdPoint.x
        );

        confidences.push_back(
            static_cast<float>(finalConfidence)
        );

        boxes.push_back(box);
    }

    // Non-Maximum Suppression
    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        confThreshold,
        nmsThreshold,
        indices
    );

    for (int index : indices) {

        Detection detection;

        detection.classId =
            classIds[index];

        detection.confidence =
            confidences[index];

        detection.box =
            boxes[index];

        detections.push_back(
            detection
        );
    }

    return detections;
}
