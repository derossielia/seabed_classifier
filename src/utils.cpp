// MAIN AUTHOR: Elia De Rossi (member A)

#include "utils.hpp"
#include <iostream>

namespace Utils {

bool loadImage(const std::string& filepath, cv::Mat& outputImage, int flags) {
    outputImage = cv::imread(filepath, flags);
    if (outputImage.empty()) {
        std::cerr << "[Error] Could not load image from: " << filepath << std::endl;
        return false;
    }
    return true;
}

double computeIoU(const cv::Mat& predMask, const cv::Mat& gtMask) {
    if (predMask.empty() || gtMask.empty()) {
        std::cerr << "[Error] Empty mask provided to computeIoU." << std::endl;
        return 0.0;
    }

    if (predMask.size() != gtMask.size() || predMask.type() != gtMask.type()) {
        std::cerr << "[Error] Mask size or type mismatch in computeIoU." << std::endl;
        return 0.0;
    }

    // Binary intersection and union
    cv::Mat intersectionMat, unionMat;
    cv::bitwise_and(predMask, gtMask, intersectionMat);
    cv::bitwise_or(predMask, gtMask, unionMat);

    int intersectionCount = cv::countNonZero(intersectionMat);
    int unionCount = cv::countNonZero(unionMat);

    if (unionCount == 0) {
        return 1.0; // Both masks are completely empty
    }

    return static_cast<double>(intersectionCount) / static_cast<double>(unionCount);
}

double computePixelAccuracy(const cv::Mat& predMask, const cv::Mat& gtMask) {
    if (predMask.empty() || gtMask.empty()) {
        std::cerr << "[Error] Empty mask provided to computePixelAccuracy." << std::endl;
        return 0.0;
    }

    if (predMask.size() != gtMask.size()) {
        std::cerr << "[Error] Size mismatch in computePixelAccuracy." << std::endl;
        return 0.0;
    }

    cv::Mat matches;
    cv::compare(predMask, gtMask, matches, cv::CMP_EQ);

    int correctPixels = cv::countNonZero(matches);
    int totalPixels = predMask.rows * predMask.cols;

    return static_cast<double>(correctPixels) / static_cast<double>(totalPixels);
}

cv::Mat blendOverlay(const cv::Mat& baseImage, const cv::Mat& mask, const cv::Scalar& color, double alpha) {
    if (baseImage.empty()) {
        return cv::Mat();
    }

    cv::Mat coloredLayer = cv::Mat(baseImage.size(), baseImage.type(), color);
    cv::Mat blended;
    cv::addWeighted(coloredLayer, alpha, baseImage, 1.0 - alpha, 0, blended);

    cv::Mat result = baseImage.clone();
    blended.copyTo(result, mask);
    return result;
}

} // namespace Utils
