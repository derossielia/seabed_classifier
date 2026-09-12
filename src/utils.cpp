// MAIN AUTHOR: Elia De Rossi (member A)

#include "utils.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include <set>

namespace Utils {

bool loadImage(const std::string& filepath, cv::Mat& outputImage) {
    outputImage = cv::imread(filepath, cv::IMREAD_COLOR);
    return !outputImage.empty();
}

bool ensureDirectoryExists(const std::string& folderPath) {
    struct stat info;
    if (stat(folderPath.c_str(), &info) != 0) {
        return mkdir(folderPath.c_str(), 0755) == 0;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

bool savePredictionTxt(const std::string& outputDir, const std::string& originalFilename, const std::string& label) {
    // Strip original directory and extension
    size_t lastSlash = originalFilename.find_last_of("/\\");
    std::string base = (lastSlash == std::string::npos) ? originalFilename : originalFilename.substr(lastSlash + 1);
    
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
    // Bottom-left positioning
    cv::Point origin(20, image.rows - 25);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.9;
    int thickness = 2;

    // Dark contour for legibility against murky backdrops
    cv::putText(image, label, origin, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(image, label, origin, fontFace, fontScale, cv::Scalar(0, 255, 255), thickness);
}

EvaluationSummary evaluate(const std::vector<std::string>& groundTruths,
                          const std::vector<std::string>& predictions) {
    EvaluationSummary summary;
    if (groundTruths.empty() || groundTruths.size() != predictions.size()) {
        return summary;
    }

    int totalSamples = static_cast<int>(groundTruths.size());
    int totalCorrect = 0;
    std::set<std::string> classes;

    for (size_t i = 0; i < groundTruths.size(); ++i) {
        classes.insert(groundTruths[i]);
        classes.insert(predictions[i]);
        summary.confusionMatrix[groundTruths[i]][predictions[i]]++;
        if (groundTruths[i] == predictions[i]) {
            totalCorrect++;
        }
    }

    summary.overallAccuracy = static_cast<double>(totalCorrect) / totalSamples;

    for (const auto& cls : classes) {
        int tp = summary.confusionMatrix[cls][cls];
        int fp = 0;
        int fn = 0;

        for (const auto& other : classes) {
            if (other != cls) {
                fp += summary.confusionMatrix[other][cls];
                fn += summary.confusionMatrix[cls][other];
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