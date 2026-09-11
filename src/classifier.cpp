// MAIN AUTHOR: Marco Rossi (member B)

#include "classifier.hpp"
#include <opencv2/opencv.hpp>

Classifier::Classifier() {}

std::string Classifier::predict(const cv::Mat& inputImage) {
    if (inputImage.empty()) return "Bare soil";

    cv::Mat hsv, gray;
    cv::cvtColor(inputImage, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);

    // 1. Vegetation detector: True seagrass has distinct saturation and hue
    cv::Mat greenMask;
    // Stricter lower bound on saturation (S >= 60) avoids pale green water turbidity
    cv::inRange(hsv, cv::Scalar(32, 60, 20), cv::Scalar(86, 255, 255), greenMask);
    double greenFraction = static_cast<double>(cv::countNonZero(greenMask)) / (gray.rows * gray.cols);

    // 2. Texture & edge statistics for Stone detection
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar meanVal, stdDevVal;
    cv::meanStdDev(laplacian, meanVal, stdDevVal);
    double laplacianVariance = stdDevVal[0] * stdDevVal[0];

    cv::Mat edges;
    cv::Canny(gray, edges, 70, 160);
    double edgeFraction = static_cast<double>(cv::countNonZero(edges)) / (gray.rows * gray.cols);

    // Decision hierarchy
    if (greenFraction > 0.12) {
        return "Vegetation";
    } else if (laplacianVariance > 95.0 || edgeFraction > 0.032) {
        return "Stones";
    } else {
        return "Bare soil";
    }
}