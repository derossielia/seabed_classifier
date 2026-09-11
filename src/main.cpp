// MAIN AUTHOR: Marco Rossi (member B)

#include "preprocessing.hpp"
#include "classifier.hpp"
#include "utils.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // 1. Validate arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_dir> [output_dir]" << std::endl;
        return -1;
    }

    std::string datasetDir = argv[1];
    std::string outputDir = (argc >= 3) ? argv[2] : "./output";

    Utils::ensureDirectoryExists(outputDir);

    const std::vector<std::string> categories = {"Bare soil", "Stones", "Vegetation"};
    std::vector<std::string> allGroundTruths;
    std::vector<std::string> allPredictions;

    // 2. Iterate through categories
    for (const auto& cat : categories) {
        fs::path catPath = fs::path(datasetDir) / cat;
        if (!fs::exists(catPath)) continue;

        for (const auto& file : fs::directory_iterator(catPath)) {
            if (file.path().extension() == ".png" || file.path().extension() == ".jpg") {
                cv::Mat img;
                if (!Utils::loadImage(file.path().string(), img)) continue;

                // Preprocess & Predict
                cv::Mat denoised = Preprocessing::removeNoise(img);
                cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
                std::string predictedLabel = Classifier::predict(enhanced);

                allGroundTruths.push_back(cat);
                allPredictions.push_back(predictedLabel);

                // Save .txt prediction into output folder
                Utils::savePredictionTxt(outputDir, file.path().filename().string(), predictedLabel);

                // Overlay label and save annotated image to output folder
                cv::Mat annotated = img.clone();
                Utils::overlayLabel(annotated, predictedLabel);
                std::string outImgPath = (fs::path(outputDir) / file.path().filename()).string();
                cv::imwrite(outImgPath, annotated);
            }
        }
    }

    // 3. Print Evaluation Summary Table for the Report
    Utils::EvaluationSummary summary = Utils::evaluate(allGroundTruths, allPredictions);
    std::cout << "\n================ EVALUATION SUMMARY ================\n";
    std::cout << "Overall Accuracy: " << summary.overallAccuracy * 100.0 << "%\n\n";

    for (const auto& [cls, cm] : summary.perClass) {
        std::cout << "Class: " << cls << "\n";
        std::cout << "  Precision: " << cm.precision * 100.0 << "%\n";
        std::cout << "  Recall:    " << cm.recall * 100.0 << "%\n";
        std::cout << "  F1-Score:  " << cm.f1Score * 100.0 << "%\n\n";
    }

    return 0;
}