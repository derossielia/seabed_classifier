// src/utils.cpp
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

namespace Utils {

bool createDirectory(const std::string& path) {
    #if defined(_WIN32)
        int res = _mkdir(path.c_str());
    #else
        int res = mkdir(path.c_str(), 0777);
    #endif
    return (res == 0 || errno == EEXIST);
}

std::string getFileStem(const std::string& filepath) {
    size_t sep = filepath.find_last_of("/\\");
    std::string filename = (sep == std::string::npos) ? filepath : filepath.substr(sep + 1);
    size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos) return filename;
    return filename.substr(0, dot);
}

bool loadImage(const std::string& filepath, cv::Mat& outputImage) {
    outputImage = cv::imread(filepath, cv::IMREAD_COLOR);
    return !outputImage.empty();
}

bool savePredictionTxt(const std::string& outputDir, const std::string& filenameStem, const std::string& label) {
    std::string outPath = outputDir + "/" + filenameStem + ".txt";
    std::ofstream out(outPath.c_str());
    if (!out.is_open()) return false;
    out << label << "\n";
    out.close();
    return true;
}

void overlayLabel(cv::Mat& image, const std::string& label) {
    if (image.empty()) return;
    cv::Point textOrigin(20, image.rows - 20);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;

    cv::putText(image, label, textOrigin, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(image, label, textOrigin, fontFace, fontScale, cv::Scalar(0, 255, 255), thickness);
}

// ---> THIS DEFINITION WAS MISSING OR ACCIDENTALLY REMOVED <---
EvaluationSummary evaluate(const std::vector<std::string>& groundTruths, const std::vector<std::string>& predictions) {
    EvaluationSummary summary;
    if (groundTruths.empty() || groundTruths.size() != predictions.size()) {
        return summary;
    }

    int correct = 0;
    int total = static_cast<int>(groundTruths.size());

    // Build confusion matrix & calculate accuracy
    for (size_t i = 0; i < groundTruths.size(); ++i) {
        const std::string& gt = groundTruths[i];
        const std::string& pred = predictions[i];

        summary.confusionMatrix[gt][pred]++;
        if (gt == pred) {
            correct++;
        }
    }
    summary.overallAccuracy = static_cast<double>(correct) / total;

    // Calculate precision, recall, and F1-score per class
    for (const auto& row : summary.confusionMatrix) {
        const std::string& cls = row.first;

        int tp = 0;
        int fn = 0;
        int fp = 0;

        // True Positives
        if (summary.confusionMatrix.count(cls) && summary.confusionMatrix[cls].count(cls)) {
            tp = summary.confusionMatrix[cls][cls];
        }

        // False Negatives (GT is this class, but predicted as something else)
        for (const auto& col : row.second) {
            if (col.first != cls) {
                fn += col.second;
            }
        }

        // False Positives (GT is another class, but predicted as this class)
        for (const auto& otherRow : summary.confusionMatrix) {
            if (otherRow.first != cls) {
                if (otherRow.second.count(cls)) {
                    fp += otherRow.second.at(cls);
                }
            }
        }

        ClassMetrics metrics;
        metrics.precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        metrics.recall    = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        metrics.f1Score   = (metrics.precision + metrics.recall > 0.0)
                                ? 2.0 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
                                : 0.0;

        summary.perClass[cls] = metrics;
    }

    return summary;
}

} // namespace Utils