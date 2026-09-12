// MAIN AUTHOR: Elia De Rossi (member A)

#include "preprocessing.hpp"
#include <vector>

namespace Preprocessing {

cv::Mat removeNoise(const cv::Mat& inputImage, int kernelSize) {
    if (inputImage.empty()) return cv::Mat();
    if (kernelSize % 2 == 0) kernelSize += 1;
    cv::Mat denoised;
    cv::medianBlur(inputImage, denoised, kernelSize);
    return denoised;
}

cv::Mat convertToHSV(const cv::Mat& inputImage) {
    if (inputImage.empty()) return cv::Mat();
    cv::Mat hsv;
    cv::cvtColor(inputImage, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

cv::Mat enhanceContrast(const cv::Mat& inputImage) {
    if (inputImage.empty()) return cv::Mat();

    cv::Mat lab;
    cv::cvtColor(inputImage, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // Clip limit 1.5 enhances without creating phantom rock contours on silt
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::Mat enhancedLab;
    cv::merge(channels, enhancedLab);

    cv::Mat result;
    cv::cvtColor(enhancedLab, result, cv::COLOR_Lab2BGR);
    return result;
}

} // namespace Preprocessing