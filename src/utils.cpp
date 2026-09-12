// MAIN AUTHOR: Elia De Rossi (member A)

#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <set>

namespace Utils {

bool loadImage(const std::string& filepath, cv::Mat& outputImage) {
    outputImage = cv::imread(filepath, cv::IMREAD_COLOR);
    return !outputImage.empty();
}

bool savePredictionTxt(const std::string& imagePath, const std::string& label) {
    // Replace file extension with .txt
    size_t lastDot = imagePath.find_last_of(".");
    std::string txtPath = (lastDot == std::string::npos) ? (imagePath + ".txt") : (imagePath.substr(0, lastDot) + ".txt");

    std::ofstream out(txtPath);
    if (!out.is_open()) return false;
    out << label << "\n";
    out.close();
    return true;
}

void overlayLabel(cv::Mat& image, const std::string& label) {
    if (image.empty()) return;
    // Bottom-left corner: origin (x=20, y=height-20)
    cv::Point textOrigin(20, image.rows - 20);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;

    // Background shadow for contrast against murky water
    cv::putText(image, label, textOrigin, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(image, label, textOrigin, fontFace, fontScale, cv::Scalar(0, 255, 255), thickness);
}

EvaluationSummary evaluate(const std::vector<std::string>& groundTruths,
                          const std::vector<std::string>& predictions) {
    EvaluationSummary summary;
    if (groundTruths.empty() || groundTruths.size() != predictions.size()) {
        return summary;
    }

    int totalSamples = static_cast<int>(groundTruths.size());
    int totalCorrect = 0;

    // Unique classes in dataset
    std::set<std::string> classes;
    for (size_t i = 0; i < groundTruths.size(); ++i) {
        classes.insert(groundTruths[i]);
        classes.insert(predictions[i]);
        summary.confusionMatrix[groundTruths[i]][predictions[i]]++;
        if (groundTruths[i] == predictions[i]) {
            totalCorrect++;
        }
    }

    // Overall Accuracy (0-1 Loss)
    summary.overallAccuracy = static_cast<double>(totalCorrect) / totalSamples;

    // Per-class Precision, Recall, F1
    for (const auto& cls : classes) {
        int tp = summary.confusionMatrix[cls][cls];
        int fp = 0;
        int fn = 0;

        for (const auto& other : classes) {
            if (other != cls) {
                fp += summary.confusionMatrix[other][cls]; // Predicted cls, but actually other
                fn += summary.confusionMatrix[cls][other]; // Actually cls, but predicted other
            }
        }

        ClassMetrics cm;
        cm.precision = (tp + fp > 0) ? (static_cast<double>(tp) / (tp + fp)) : 0.0;
        cm.recall    = (tp + fn > 0) ? (static_cast<double>(tp) / (tp + fn)) : 0.0;
        cm.f1Score   = (cm.precision + cm.recall > 0.0) 
                     ? (2.0 * cm.precision * cm.recall / (cm.precision + cm.recall)) 
                     : 0.0;

        summary.perClass[cls] = cm;
    }

    return summary;
}

} // namespace Utils
