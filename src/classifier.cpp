// MAIN AUTHOR: Marco Rossi (member B)

#include "classifier.hpp"
#include <opencv2/opencv.hpp>

Classifier::Classifier() {}

std::string Classifier::predict(const cv::Mat& inputImage) {
    if (inputImage.empty()) return "Bare soil";

    cv::Mat hsv, gray;
    cv::cvtColor(inputImage, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);

    // 1. Vegetation Feature: Detect seagrass/algae patches
    // OpenCV Hue: Green is [35, 85]. Require minimum saturation to avoid green water haze.
    cv::Mat greenMask;
    cv::inRange(hsv, cv::Scalar(32, 40, 20), cv::Scalar(88, 255, 255), greenMask);
    double greenFraction = static_cast<double>(cv::countNonZero(greenMask)) / (gray.rows * gray.cols);

    // 2. Stone Feature: Laplacian variance measures high-frequency structural roughness
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar meanVal, stdDevVal;
    cv::meanStdDev(laplacian, meanVal, stdDevVal);
    double textureVariance = stdDevVal[0] * stdDevVal[0];

    // 3. Stone Feature: Distinct boundary edges
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 130);
    double edgeFraction = static_cast<double>(cv::countNonZero(edges)) / (gray.rows * gray.cols);

    // Decision logic using physical seabed properties
    if (greenFraction > 0.08) {
        return "Vegetation";
    } else if (textureVariance > 45.0 || edgeFraction > 0.015) {
        return "Stones";
    } else {
        return "Bare soil";
    }
}