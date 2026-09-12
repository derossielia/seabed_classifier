// MAIN AUTHOR: Marco Rossi (member B)
#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Classifier {
public:
    Classifier();

    // Predicts the benthic habitat class ("Bare soil", "Stones", "Vegetation")
    std::string predict(const cv::Mat& inputImage);

private:
    // Extracts average HSV color characteristics
    cv::Scalar extractHSVFeatures(const cv::Mat& hsvImage);

    // Computes edge pixel density using Sobel gradient
    double extractEdgeDensity(const cv::Mat& grayImage);
};

#endif // CLASSIFIER_HPP