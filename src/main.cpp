// MAIN AUTHOR: Marco Rossi (member B)
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "classifier.hpp"
#include "preprocessing.hpp"
#include "utils.hpp"

cv::Mat createMaskedVisualization(const cv::Mat& preprocessedImg, 
                                 const cv::Mat& rockMask, 
                                 const cv::Mat& vegMask) 
{
    cv::Mat overlay = preprocessedImg.clone();

    // Vegetation colored in Green (B=0, G=255, R=0)
    overlay.setTo(cv::Scalar(0, 255, 0), vegMask);

    // Rocks colored in Red (B=0, G=0, R=255)
    overlay.setTo(cv::Scalar(0, 0, 255), rockMask);

    // Alpha blend: 65% original image + 35% colored mask
    cv::Mat blended;
    cv::addWeighted(overlay, 0.35, preprocessedImg, 0.65, 0, blended);

    return blended;
}

cv::Mat preprocessedImage;

int main(int argc, char** argv) {
    std::string datasetPath = "../data";

    if (argc > 1 && argv[1] != nullptr) {
        std::string candidate = argv[1];
        if (candidate.find("=") == std::string::npos && !candidate.empty()) {
            datasetPath = candidate;
        }
    }

    std::cout << "[INFO] Loading dataset from: " << datasetPath << std::endl;

    std::string textOutputDir = "output_labels";
    std::string imagesOutputDir = "output_images";

    Utils::createDirectory(textOutputDir);
    Utils::createDirectory(imagesOutputDir);

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
            cv::Mat denoised = Preprocessing::removeNoise(rawImage, 3);
            cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
            std::string predictedLabel = classifier.predict(enhanced);

            cv::Mat hsv, gray, edges;
            cv::Mat vegMask, rockMask;

            // Extract masks corresponding to underwater features
            cv::cvtColor(preprocessedImage, hsv, cv::COLOR_BGR2HSV);
            // Algae/seagrass: green/brown hue range in HSV
            cv::inRange(hsv, cv::Scalar(25, 35, 20), cv::Scalar(85, 255, 255), vegMask);

            cv::cvtColor(preprocessedImage, gray, cv::COLOR_BGR2GRAY);
            cv::Canny(gray, edges, 40, 120);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(edges, rockMask, kernel);
            rockMask = rockMask & (~vegMask); // Exclude vegetation texture

            // Create the combined overlaid image
            cv::Mat maskedVisualization = createMaskedVisualization(preprocessedImage, rockMask, vegMask);

            // Display
            cv::namedWindow("Preprocessed Seabed (Overlaid Masks)", cv::WINDOW_NORMAL);
            cv::imshow("Preprocessed Seabed (Overlaid Masks)", maskedVisualization);

            // Wait for keypress to step through each frame (Press ESC or 'q' to close)
            int key = cv::waitKey(0);
            if (key == 27 || key == 'q') {
                // Optional: exit or continue
            }

            groundTruths.push_back(category);
            predictions.push_back(predictedLabel);
            std::string fileStem = Utils::getFileStem(pathStr);

            // Export results
            cv::Mat annotatedImage = rawImage.clone();
            Utils::overlayLabel(annotatedImage, predictedLabel);
            std::string imgOutPath = imagesOutputDir + "/" + fileStem + "_annotated.jpg";
            cv::imwrite(imgOutPath, annotatedImage);

            Utils::savePredictionTxt(textOutputDir, fileStem, predictedLabel);
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
    // Helper to overlay colored masks onto an image for inspection
