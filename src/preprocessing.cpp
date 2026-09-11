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
    cv::bilateralFilter(inputImage, filtered, 9, 75.0, 75.0);
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
    if (inputImage.empty()) return cv::Mat();

    cv::Mat lab;
    cv::cvtColor(inputImage, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // Lower clipLimit (1.5) prevents boosting murky particulate noise
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::Mat enhancedLab;
    cv::merge(channels, enhancedLab);

    cv::Mat result;
    cv::cvtColor(enhancedLab, result, cv::COLOR_Lab2BGR);
    return result;
}

} // namespace Preprocessing