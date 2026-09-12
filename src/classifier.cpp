// MAIN AUTHOR: Marco Rossi (member B)

#include "classifier.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>

Classifier::Classifier() {}

HabitatScores Classifier::computeScores(const cv::Mat& inputImage) {
    HabitatScores scores;
    if (inputImage.empty()) return scores;

    cv::Mat hsv, gray;
    cv::cvtColor(inputImage, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);

    int totalPixels = gray.rows * gray.cols;

    // 1. Vegetation detector: HSV mask for true seagrass/algae pigments
    cv::Mat vegMask;
    cv::inRange(hsv, cv::Scalar(32, 45, 20), cv::Scalar(86, 255, 255), vegMask);
    double vegDensity = static_cast<double>(cv::countNonZero(vegMask)) / totalPixels;
    // Normalized score: saturates at 15% green coverage
    scores.veg = std::min(1.0, vegDensity / 0.12);

    // 2. Stone detector: High spatial frequency & contour edges
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar meanVal, stdDevVal;
    cv::meanStdDev(laplacian, meanVal, stdDevVal);
    double variance = stdDevVal[0] * stdDevVal[0];

    cv::Mat edges;
    cv::Canny(gray, edges, 40, 110);
    double edgeDensity = static_cast<double>(cv::countNonZero(edges)) / totalPixels;

    // Stones have both high gradient variance and distinct contours
    double normVar = std::min(1.0, variance / 50.0);
    double normEdges = std::min(1.0, edgeDensity / 0.035);
    scores.stone = 0.5 * normVar + 0.5 * normEdges;

    // 3. Bare Soil detector: Active measurement of flatness / uniformity
    // High score when edges and variance are absent
    double roughness = std::min(1.0, (variance / 30.0) * 0.5 + (edgeDensity / 0.02) * 0.5);
    scores.soil = std::max(0.0, 1.0 - roughness);

    return scores;
}

std::string Classifier::predict(const cv::Mat& inputImage) {
    HabitatScores s = computeScores(inputImage);

    // Weights allow fine-tuning class sensitivity
    double w_veg   = 1.10 * s.veg;
    double w_stone = 1.00 * s.stone;
    double w_soil  = 0.95 * s.soil;

    if (w_veg >= w_stone && w_veg >= w_soil && s.veg > 0.15) {
        return "Vegetation";
    }
    if (w_stone >= w_soil) {
        return "Stones"; // Keep matching your folder name
    }
    return "Bare soil";
}