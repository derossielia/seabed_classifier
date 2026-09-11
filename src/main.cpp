// MAIN AUTHOR: Marco Rossi (member B)

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "classifier.hpp"
#include "preprocessing.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    std::string datasetPath = "../dataset";
    // Project root output folder (when executing from build/)
    std::string outputDir = "../output";

    if (argc > 1 && argv[1] != nullptr) {
        std::string candidate = argv[1];
        if (candidate.find("=") == std::string::npos && !candidate.empty()) {
            datasetPath = candidate;
        }
    }
    if (argc > 2 && argv[2] != nullptr) {
        outputDir = argv[2];
    }

    Utils::ensureDirectoryExists(outputDir);

    std::cout << "[INFO] Loading dataset from: " << datasetPath << std::endl;
    std::cout << "[INFO] Storing results into: " << outputDir << std::endl;

    Classifier classifier;
    std::vector<std::string> groundTruths;
    std::vector<std::string> predictions;

    const std::vector<std::string> categories = {"Bare soil", "Stones", "Vegetation"};

    for (const auto& category : categories) {
        std::string categoryDir = datasetPath + "/" + category;
        std::vector<cv::String> filepaths;
        
        cv::glob(categoryDir + "/*.*", filepaths, false);

        int validImageCount = 0;
        for (const auto& filepath : filepaths) {
            std::string pathStr = filepath;
            if (pathStr.find(".txt") != std::string::npos || 
                pathStr.find(".DS_Store") != std::string::npos) {
                continue;
            }

            cv::Mat rawImage;
            if (!Utils::loadImage(pathStr, rawImage)) {
                continue;
            }

            validImageCount++;

            // Pipeline: Clean -> Contrast -> Predict
            cv::Mat denoised = Preprocessing::removeNoise(rawImage, 5);
            cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
            std::string predictedLabel = classifier.predict(enhanced);

            groundTruths.push_back(category);
            predictions.push_back(predictedLabel);

            // Save .txt prediction into root output/
            Utils::savePredictionTxt(outputDir, pathStr, predictedLabel);

            // Overlay label on bottom-left and save annotated image into root output/
            size_t lastSlash = pathStr.find_last_of("/\\");
            std::string baseFilename = (lastSlash == std::string::npos) ? pathStr : pathStr.substr(lastSlash + 1);

            cv::Mat annotatedImage = rawImage.clone();
            Utils::overlayLabel(annotatedImage, predictedLabel);
            cv::imwrite(outputDir + "/" + baseFilename, annotatedImage);
        }

        std::cout << "Category '" << category << "': found " << validImageCount << " valid images." << std::endl;
    }

    // Terminal reporting
    if (!groundTruths.empty()) {
        Utils::EvaluationSummary summary = Utils::evaluate(groundTruths, predictions);
        std::cout << "\n=== EVALUATION SUMMARY ===" << std::endl;
        std::cout << "Overall Accuracy: " << summary.overallAccuracy * 100.0 << "%\n" << std::endl;

        for (const auto& cat : categories) {
            std::cout << "--- Class: " << cat << " ---" << std::endl;
            if (summary.perClass.count(cat) > 0) {
                std::cout << "  Precision: " << summary.perClass[cat].precision * 100.0 << "%" << std::endl;
                std::cout << "  Recall:    " << summary.perClass[cat].recall * 100.0 << "%" << std::endl;
                std::cout << "  F1-Score:  " << summary.perClass[cat].f1Score * 100.0 << "%\n" << std::endl;
            } else {
                std::cout << "  No data available.\n" << std::endl;
            }
        }
    } else {
        std::cerr << "\n[Warning] No valid images found in the specified path!" << std::endl;
    }

    return 0;
}