// MAIN AUTHOR: Elia De Rossi (member A)

#include "preprocessing.hpp"

namespace Preprocessing {

cv::Mat removeNoise(const cv::Mat& inputImage, int kernelSize) {
    if (inputImage.empty()) {
        return cv::Mat();
    }

    // Ensure kernel size is an odd integer >= 3
    if (kernelSize <= 1) {
        kernelSize = 3;
    } else if (kernelSize % 2 == 0) {
        kernelSize += 1;
    }

    cv::Mat filtered;
    cv::medianBlur(inputImage, filtered, kernelSize);
    return filtered;
}

cv::Mat convertToHSV(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        return cv::Mat();
    }

    cv::Mat hsvImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);
    } else {
        hsvImage = inputImage.clone();
    }

    return hsvImage;
}

cv::Mat enhanceContrast(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        return cv::Mat();
    }

    // For multi-channel images, equalize luminance only to preserve chromaticity
    if (inputImage.channels() == 3) {
        cv::Mat ycrcb;
        cv::cvtColor(inputImage, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        // CLAHE avoids over-amplifying background underwater haze
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(channels[0], channels[0]);

        cv::Mat enhancedYCrCb;
        cv::merge(channels, enhancedYCrCb);

        cv::Mat result;
        cv::cvtColor(enhancedYCrCb, result, cv::COLOR_YCrCb2BGR);
        return result;
    }

    // Single-channel fallback
    cv::Mat result;
    cv::equalizeHist(inputImage, result);
    return result;
}

} // namespace Preprocessing