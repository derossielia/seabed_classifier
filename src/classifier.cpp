// MAIN AUTHOR: Marco Rossi (member B)

#include "classifier.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

Classifier::Classifier() {}

double Classifier::extractGreenFraction(const cv::Mat& hsvImage) {
    if (hsvImage.empty()) return 0.0;
    // Green seagrass: Hue in [32, 85]. Saturation >= 45 avoids murky green water tint
    cv::Mat greenMask;
    cv::inRange(hsvImage, cv::Scalar(32, 45, 20), cv::Scalar(86, 255, 255), greenMask);
    return static_cast<double>(cv::countNonZero(greenMask)) / (hsvImage.rows * hsvImage.cols);
}

double Classifier::extractTextureVariance(const cv::Mat& grayImage) {
    if (grayImage.empty()) return 0.0;
    cv::Mat laplacian;
    cv::Laplacian(grayImage, laplacian, CV_64F);
    cv::Scalar meanVal, stdDevVal;
    cv::meanStdDev(laplacian, meanVal, stdDevVal);
    return stdDevVal[0] * stdDevVal[0];
}

double Classifier::extractCannyEdgeDensity(const cv::Mat& grayImage) {
    if (grayImage.empty()) return 0.0;
    cv::Mat edges;
    // Lower thresholds (30, 90) to capture soft submerged rock boundaries
    cv::Canny(grayImage, edges, 30, 90);
    return static_cast<double>(cv::countNonZero(edges)) / (grayImage.rows * grayImage.cols);
}

std::string Classifier::predict(const cv::Mat& inputImage) {
    if (inputImage.empty()) return "Bare soil";

    cv::Mat hsvImage, grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_GRAY2BGR);
        cv::cvtColor(hsvImage, hsvImage, cv::COLOR_BGR2HSV);
    }

    double greenFraction   = extractGreenFraction(hsvImage);
    double textureVariance = extractTextureVariance(grayImage);
    double edgeDensity     = extractCannyEdgeDensity(grayImage);

    // 1. Vegetation: dense organic patches
    if (greenFraction > 0.09) {
        return "Vegetation";
    }
    // 2. Stone: submerged rocks produce noticeable local roughness and contours
    else if (textureVariance > 18.0 || edgeDensity > 0.012) {
        return "Stones"; // Matches folder name; main.cpp will normalize "Stone" vs "Stones"
    }
    // 3. Bare soil: smooth mud/sand silt
    else {
        return "Bare soil";
    }
}