// MAIN AUTHOR: Marco Rossi (member B)
#include "classifier.hpp"

Classifier::Classifier() {}

// Extracts average HSV channel values
cv::Scalar Classifier::extractHSVFeatures(const cv::Mat& hsvImage) {
    if (hsvImage.empty()) {
        return cv::Scalar(0, 0, 0);
    }
    return cv::mean(hsvImage);
}

// Computes edge pixel density using Sobel gradients
double Classifier::extractEdgeDensity(const cv::Mat& grayImage) {
    if (grayImage.empty()) {
        return 0.0;
    }

    cv::Mat gradX, gradY, absGradX, absGradY, grad, binaryEdges;

    // Gradient calculation
    cv::Sobel(grayImage, gradX, CV_16S, 1, 0);
    cv::Sobel(grayImage, gradY, CV_16S, 0, 1);
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);

    // Edge binarization
    cv::threshold(grad, binaryEdges, 50, 255, cv::THRESH_BINARY);

    int edgePixels = cv::countNonZero(binaryEdges);
    return static_cast<double>(edgePixels) / (grayImage.rows * grayImage.cols);
}

// Predicts the benthic habitat class based on color, texture, and intensity dispersion
std::string Classifier::predict(const cv::Mat& inputImage) {
    if (inputImage.empty()) return "Unknown";

    cv::Mat hsv, gray;
    cv::cvtColor(inputImage, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);

    // 1. Color Criterion: Percentage of distinct green vegetation pixels
    // Hue in OpenCV: [35, 85] covers yellowish-green to deep green
    cv::Mat greenMask;
    cv::inRange(hsv, cv::Scalar(35, 45, 30), cv::Scalar(85, 255, 255), greenMask);
    double greenPixelFraction = static_cast<double>(cv::countNonZero(greenMask)) / (gray.rows * gray.cols);

    // 2. Texture Criterion: Laplacian variance measures structural roughness
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar meanVal, stdDevVal;
    cv::meanStdDev(laplacian, meanVal, stdDevVal);
    double textureVariance = stdDevVal[0] * stdDevVal[0]; // Variance of gradients

    // 3. Edge Contrast Criterion (Strong structural stone contours)
    cv::Mat cannyEdges;
    cv::Canny(gray, cannyEdges, 60, 150);
    double strongEdgeDensity = static_cast<double>(cv::countNonZero(cannyEdges)) / (gray.rows * gray.cols);

    // Decision Logic based on physical features
    if (greenPixelFraction > 0.20) {
        return "Vegetation";
    } else if (strongEdgeDensity > 0.025 || textureVariance > 85.0) {
        return "Stones";
    } else {
        return "Bare soil";
    }
}