// MAIN AUTHOR: Elia De Rossi (member A)

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
    // Return true if directory was created or already exists
    return (res == 0 || errno == EEXIST);
}

std::string getFileStem(const std::string& filepath) {
    // Strip leading path directories
    size_t sep = filepath.find_last_of("/\\");
    std::string filename = (sep == std::string::npos) ? filepath : filepath.substr(sep + 1);

    // Strip extension
    size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos) {
        return filename;
    }
    return filename.substr(0, dot);
}

bool loadImage(const std::string& filepath, cv::Mat& outputImage) {
    outputImage = cv::imread(filepath, cv::IMREAD_COLOR);
    return !outputImage.empty();
}

bool savePredictionTxt(const std::string& outputDir, const std::string& filename, const std::string& label) {
    size_t lastSlash = filename.find_last_of("/\\");
    std::string base = (lastSlash == std::string::npos) ? filename : filename.substr(lastSlash + 1);
    
    size_t lastDot = base.find_last_of(".");
    std::string stem = (lastDot == std::string::npos) ? base : base.substr(0, lastDot);

    std::string outPath = outputDir + "/" + stem + ".txt";
    std::ofstream out(outPath.c_str());
    if (!out.is_open()) {
        std::cerr << "Failed to open " << outPath << " for writing." << std::endl;
        return false;
    }
    out << label << "\n";
    out.close();
    return true;
}


void overlayLabel(cv::Mat& image, const std::string& label) {
    if (image.empty()) return;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    int thickness = 2;
    cv::Point org(20, image.rows - 20);

    // Black outline for contrast, followed by green/yellow text
    cv::putText(image, label, org, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(image, label, org, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
}

EvaluationSummary evaluate(const std::vector<std::string>& groundTruths, const std::vector<std::string>& predictions) {
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

        Utils::ClassMetrics cm;
        cm.precision = (tp + fp > 0) ? (static_cast<double>(tp) / (tp + fp)) : 0.0;
        cm.recall    = (tp + fn > 0) ? (static_cast<double>(tp) / (tp + fn)) : 0.0;
        cm.f1Score   = (cm.precision + cm.recall > 0.0) 
                     ? (2.0 * cm.precision * cm.recall / (cm.precision + cm.recall)) 
                     : 0.0;

        summary.perClass[cls] = cm;
    }

    return summary;
    }
}

 // namespace Utils