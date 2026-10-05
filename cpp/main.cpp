#include "detector.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    const std::string modelPath =
        "models/yolov5n.onnx";

    YOLODetector detector(
        modelPath,
        0.4f,
        0.45f
    );

    cv::VideoCapture camera(0);

    if (!camera.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return 1;
    }

    camera.set(
        cv::CAP_PROP_FRAME_WIDTH,
        640
    );

    camera.set(
        cv::CAP_PROP_FRAME_HEIGHT,
        480
    );

    std::cout << "Camera started\n";
    std::cout << "Press Q to quit\n";

    cv::Mat frame;

    auto previousTime =
        std::chrono::steady_clock::now();

    while (true) {

        camera >> frame;

        if (frame.empty()) {
            std::cerr
                << "Empty camera frame\n";
            break;
        }

        // YOLO
        std::vector<Detection> detections =
            detector.detect(frame);

        // 畫 Detection
        for (const auto& detection : detections) {

            cv::rectangle(
                frame,
                detection.box,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );

            std::string label =
                "class " +
                std::to_string(
                    detection.classId
                ) +
                " " +
                std::to_string(
                    static_cast<int>(
                        detection.confidence * 100
                    )
                ) +
                "%";

            int textY =
                std::max(
                    detection.box.y - 10,
                    20
                );

            cv::putText(
                frame,
                label,
                cv::Point(
                    detection.box.x,
                    textY
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );
        }

        // FPS
        auto currentTime =
            std::chrono::steady_clock::now();

        double elapsed =
            std::chrono::duration<double>(
                currentTime -
                previousTime
            ).count();

        previousTime =
            currentTime;

        double fps =
            1.0 / elapsed;

        cv::putText(
            frame,
            "FPS: " +
                std::to_string(
                    static_cast<int>(fps)
                ),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );

        cv::imshow(
            "Pi AI Camera - OpenCV",
            frame
        );

        char key =
            static_cast<char>(
                cv::waitKey(1)
            );

        if (key == 'q' ||
            key == 'Q') {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}
