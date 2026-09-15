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
    cv::threshold(grad, binaryEdges, 57, 255, cv::THRESH_BINARY);

    int edgePixels = cv::countNonZero(binaryEdges);
    return static_cast<double>(edgePixels) / (grayImage.rows * grayImage.cols);
}

// Predicts the benthic habitat class based on color, texture, and intensity dispersion
std::string Classifier::predict(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        return "Unknown";
    }

    cv::Mat hsvImage, grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_GRAY2BGR);
        cv::cvtColor(hsvImage, hsvImage, cv::COLOR_BGR2HSV);
    }

    cv::Scalar hsvMean = extractHSVFeatures(hsvImage);
    double edgeDensity = extractEdgeDensity(grayImage);

    double hue = hsvMean[0]; // Channel 0: Hue
    double sat = hsvMean[1]; // Channel 1: Saturation

    // Habitat classification logic
    // 1. Vegetazione: Tonalita' verde alta (Hue > 60)
    if (hue > 60.0) {
        return "Vegetation";
    } 
    // 2. Pietre: alta densita' di bordi OPPURE saturazione elevata
    else if (edgeDensity > 0.0022 || sat > 55.0) {
        return "Stones";
    } 
    // 3. Suolo nudo: superficie liscia e omogenea
    else {
        return "Bare soil";
    }
}