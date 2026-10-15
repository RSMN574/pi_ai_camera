
#include "object_tracker.hpp"

#include <algorithm>

bool ObjectTracker::select(
    const cv::Mat& frame,
    const cv::Rect& box,
    int classId
) {
    if (frame.empty() || box.width <= 0 || box.height <= 0) {
        return false;
    }

    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    const cv::Rect clipped = box & bounds;

    if (clipped.width < 2 || clipped.height < 2) {
        return false;
    }

    try {
        tracker_ = cv::TrackerKCF::create();

        if (tracker_.empty()) {
            return false;
        }

        tracker_->init(frame, clipped);

        box_ = clipped;
        lastBox_ = clipped;
        classId_ = classId;
        state_ = TrackingState::TRACKING;

        return true;
    } catch (const cv::Exception&) {
        tracker_.release();
        state_ = TrackingState::LOST;
        return false;
    }
}

bool ObjectTracker::update(const cv::Mat& frame) {
    if (state_ != TrackingState::TRACKING ||
        tracker_.empty() ||
        frame.empty()) {
        return false;
    }

    cv::Rect updatedBox = box_;

    try {
        if (!tracker_->update(frame, updatedBox)) {
            lastBox_ = box_;
            state_ = TrackingState::LOST;
            tracker_.release();
            return false;
        }
    } catch (const cv::Exception&) {
        lastBox_ = box_;
        state_ = TrackingState::LOST;
        tracker_.release();
        return false;
    }

    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    updatedBox &= bounds;

    if (updatedBox.width < 2 || updatedBox.height < 2) {
        lastBox_ = box_;
        state_ = TrackingState::LOST;
        tracker_.release();
        return false;
    }

    box_ = updatedBox;
    lastBox_ = updatedBox;

    return true;
}

void ObjectTracker::reset() {
    tracker_.release();
    box_ = cv::Rect();
    lastBox_ = cv::Rect();
    classId_ = -1;
    state_ = TrackingState::WAITING;
}

TrackingState ObjectTracker::state() const {
    return state_;
}

const cv::Rect& ObjectTracker::box() const {
    return box_;
}

const cv::Rect& ObjectTracker::lastBox() const {
    return lastBox_;
}

int ObjectTracker::classId() const {
    return classId_;
}
