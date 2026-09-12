// MAIN AUTHOR: Marco Rossi (member B)

#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <string>

struct HabitatScores {
    double soil = 0.0;
    double stone = 0.0;
    double veg = 0.0;
};

class Classifier {
public:
    Classifier();
    
    // Computes independent continuous scores for each backdrop
    HabitatScores computeScores(const cv::Mat& inputImage);

    // Predicts the class with the maximum weighted score
    std::string predict(const cv::Mat& inputImage);
};

#endif // CLASSIFIER_HPP