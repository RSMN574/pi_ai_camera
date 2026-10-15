
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <memory>

enum class TrackingState {
    WAITING,
    TRACKING,
    LOST
};

class ObjectTracker {
public:
    ObjectTracker() = default;

    bool select(
        const cv::Mat& frame,
        const cv::Rect& box,
        int classId
    );

    bool update(const cv::Mat& frame);

    void reset();

    TrackingState state() const;
    const cv::Rect& box() const;
    const cv::Rect& lastBox() const;
    int classId() const;

private:
    cv::Ptr<cv::TrackerKCF> tracker_;

    TrackingState state_ = TrackingState::WAITING;

    cv::Rect box_;
    cv::Rect lastBox_;

    int classId_ = -1;
};
