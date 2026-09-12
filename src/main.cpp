// MAIN AUTHOR: Marco Rossi (member B)

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "classifier.hpp"
#include "preprocessing.hpp"
#include "utils.hpp"

// Normalizes singular/plural so evaluation doesn't fail on string naming
std::string normalizeCategory(const std::string& label) {
    if (label == "Stone" || label == "Stones") return "Stones";
    return label;
}

int main(int argc, char** argv) {
    std::string datasetPath = "../data";
    std::string outputDir = "../output";

    if (argc > 1 && argv[1] != nullptr) datasetPath = argv[1];
    if (argc > 2 && argv[2] != nullptr) outputDir = argv[2];

    Utils::ensureDirectoryExists(outputDir);

    Classifier classifier;
    std::vector<std::string> groundTruths;
    std::vector<std::string> predictions;

    const std::vector<std::string> folderCategories = {"Bare soil", "Stones", "Vegetation"};

    for (const auto& category : folderCategories) {
        std::string categoryDir = datasetPath + "/" + category;
        std::vector<cv::String> filepaths;
        cv::glob(categoryDir + "/*.*", filepaths, false);

        int validImages = 0;
        for (const auto& filepath : filepaths) {
            std::string pathStr = filepath;
            if (pathStr.find(".png") == std::string::npos && pathStr.find(".jpg") == std::string::npos) {
                continue;
            }

            cv::Mat rawImg;
            if (!Utils::loadImage(pathStr, rawImg)) continue;
            validImages++;

            cv::Mat denoised = Preprocessing::removeNoise(rawImg, 5);
            cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
            std::string pred = classifier.predict(enhanced);

            groundTruths.push_back(normalizeCategory(category));
            predictions.push_back(normalizeCategory(pred));

            // Save .txt prediction in root output/ folder
            Utils::savePredictionTxt(outputDir, pathStr, pred);

            // Overlay label and save annotated image
            cv::Mat annotated = rawImg.clone();
            Utils::overlayLabel(annotated, pred);
            size_t slash = pathStr.find_last_of("/\\");
            std::string fname = (slash == std::string::npos) ? pathStr : pathStr.substr(slash + 1);
            cv::imwrite(outputDir + "/" + fname, annotated);
        }
        std::cout << "Category '" << category << "': found " << validImages << " valid images." << std::endl;
    }

    if (!groundTruths.empty()) {
        Utils::EvaluationSummary summary = Utils::evaluate(groundTruths, predictions);
        std::cout << "\n=== EVALUATION SUMMARY ===" << std::endl;
        std::cout << "Overall Accuracy: " << summary.overallAccuracy * 100.0 << "%\n" << std::endl;

        for (const auto& cat : folderCategories) {
            std::string normCat = normalizeCategory(cat);
            std::cout << "--- Class: " << cat << " ---" << std::endl;
            if (summary.perClass.count(normCat) > 0) {
                std::cout << "  Precision: " << summary.perClass[normCat].precision * 100.0 << "%" << std::endl;
                std::cout << "  Recall:    " << summary.perClass[normCat].recall * 100.0 << "%" << std::endl;
                std::cout << "  F1-Score:  " << summary.perClass[normCat].f1Score * 100.0 << "%\n" << std::endl;
            }
        }
    }
    return 0;
}