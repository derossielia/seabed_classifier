// MAIN AUTHOR: Marco Rossi (member B)

#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Classifier {
public:
    Classifier();
    std::string predict(const cv::Mat& inputImage);
};

#endif // CLASSIFIER_HPP