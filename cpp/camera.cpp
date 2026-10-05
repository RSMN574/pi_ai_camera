#include <libcamera/libcamera.h>

#include <iostream>
#include <memory>

int main()
{
    using namespace libcamera;

    std::unique_ptr<CameraManager> cm =
        std::make_unique<CameraManager>();

    int ret = cm->start();

    if (ret) {
        std::cerr << "Failed to start CameraManager\n";
        return ret;
    }

    std::cout << "libcamera started\n";

    const auto &cameras = cm->cameras();

    std::cout << "Camera count: "
              << cameras.size()
              << "\n";

    for (const auto &camera : cameras) {
        std::cout
            << "Camera: "
            << camera->id()
            << "\n";
    }

    cm->stop();

    return 0;
}
