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

    // Parse command line arguments
    if (argc > 1 && argv[1] != nullptr) {
        std::string candidate = argv[1];
        if (candidate.find("=") == std::string::npos && !candidate.empty()) {
            datasetPath = candidate;
        }
    }

    std::cout << "[INFO] Loading dataset from: " << datasetPath << std::endl;

    Classifier classifier;
    std::vector<std::string> groundTruths;
    std::vector<std::string> predictions;

    std::vector<std::string> categories = {"Bare soil", "Stones", "Vegetation"};

    // Process each category directory
    for (const auto& category : categories) {
        std::string categoryDir = datasetPath + "/" + category;
        std::vector<cv::String> filepaths;
        
        cv::glob(categoryDir + "/*.*", filepaths, false);

        std::cout << "Category '" << category << "': found " << filepaths.size() << " files." << std::endl;

        for (const auto& filepath : filepaths) {
            std::string pathStr = filepath;

            // Ignore non-image files, prediction text files, and annotated outputs
            if (pathStr.find(".txt") != std::string::npos || 
                pathStr.find("_annotated") != std::string::npos ||
                pathStr.find(".DS_Store") != std::string::npos) {
                continue;
            }

            cv::Mat rawImage;
            if (!Utils::loadImage(pathStr, rawImage)) {
                std::cerr << "Failed to load: " << pathStr << std::endl;
                continue;
            }

            // Image processing and classification pipeline
            cv::Mat denoised = Preprocessing::removeNoise(rawImage, 5);
            cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
            std::string predictedLabel = classifier.predict(enhanced);

            groundTruths.push_back(category);
            predictions.push_back(predictedLabel);

            // Export results
            cv::Mat annotatedImage = rawImage.clone();
            Utils::overlayLabel(annotatedImage, predictedLabel);
            cv::imwrite(pathStr + "_annotated.jpg", annotatedImage);

            Utils::savePredictionTxt(pathStr, predictedLabel);
        }
    }

    // Performance evaluation
    if (!groundTruths.empty()) {
        Utils::EvaluationSummary summary = Utils::evaluate(groundTruths, predictions);
        std::cout << "\n=== EVALUATION SUMMARY ===" << std::endl;
        std::cout << "Overall Accuracy: " << summary.overallAccuracy * 100.0 << "%" << std::endl;

        for (const auto& cat : categories) {
            std::cout << "\n--- Class: " << cat << " ---" << std::endl;
            if (summary.perClass.count(cat) > 0) {
                std::cout << "  Precision: " << summary.perClass[cat].precision * 100.0 << "%" << std::endl;
                std::cout << "  Recall:    " << summary.perClass[cat].recall * 100.0 << "%" << std::endl;
                std::cout << "  F1-Score:  " << summary.perClass[cat].f1Score * 100.0 << "%" << std::endl;
            } else {
                std::cout << "  No data available." << std::endl;
            }
        }
    } else {
        std::cout << "\nWarning: No valid images found in the specified path!" << std::endl;
    }

    return 0;
}