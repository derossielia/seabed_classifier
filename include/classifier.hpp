// MAIN AUTHOR: Marco Rossi (member B)

#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Classifier {
public:
    Classifier();
    std::string predict(const cv::Mat& inputImage);

private:
    double extractGreenFraction(const cv::Mat& hsvImage);
    double extractTextureVariance(const cv::Mat& grayImage);
    double extractCannyEdgeDensity(const cv::Mat& grayImage);
};

#endif // CLASSIFIER_HPP