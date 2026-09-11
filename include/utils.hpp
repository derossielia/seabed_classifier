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

// Safe image loader
bool loadImage(const std::string& filepath, cv::Mat& outputImage);

// Creates output directory if it doesn't exist
bool ensureDirectoryExists(const std::string& folderPath);

// Writes predicted label to a .txt file inside the designated output directory
bool savePredictionTxt(const std::string& outputDir, const std::string& originalFilename, const std::string& label);

// Writes text onto the bottom-left corner of the image (as required by specs)
void overlayLabel(cv::Mat& image, const std::string& label);

// Computes quantitative metrics using 0-1 loss over the whole dataset
EvaluationSummary evaluate(const std::vector<std::string>& groundTruths,
                          const std::vector<std::string>& predictions);

} // namespace Utils

#endif // UTILS_HPP