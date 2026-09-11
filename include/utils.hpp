// MAIN AUTHOR: Elia De Rossi (member A)

#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

namespace Utils {

struct ClassMetrics {
    double precision = 0.0;
    double recall    = 0.0;
    double f1Score   = 0.0;
};

struct EvaluationSummary {
    double overallAccuracy = 0.0;
    std::map<std::string, ClassMetrics> perClass;
    std::map<std::string, std::map<std::string, int>> confusionMatrix;
};

// Computes Accuracy, Precision, Recall, and F1 over the dataset
EvaluationSummary evaluate(const std::vector<std::string>& groundTruths,
                          const std::vector<std::string>& predictions);

// Saves the classification label to a .txt file named as the image
bool savePredictionTxt(const std::string& imagePath, const std::string& label);

// Writes the label in the bottom-left corner of the image
void overlayLabel(cv::Mat& image, const std::string& label);

// Safe loader
bool loadImage(const std::string& filepath, cv::Mat& outputImage);

} // namespace Utils

#endif // UTILS_HPP