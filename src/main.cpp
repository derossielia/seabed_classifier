// MAIN AUTHOR: Marco Rossi (member B)

#include "preprocessing.hpp"
#include "classifier.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_dir> [output_dir]" << std::endl;
        return -1;
    }

    std::string datasetDir = argv[1];
    std::string outputDir = (argc >= 3) ? argv[2] : "./output";

    Utils::ensureDirectoryExists(outputDir);

    Classifier classifier;
    const std::vector<std::string> categories = {"Bare soil", "Stones", "Vegetation"};
    std::vector<std::string> allGroundTruths;
    std::vector<std::string> allPredictions;

    for (const auto& cat : categories) {
        std::string folderPattern = datasetDir + "/" + cat + "/*";
        std::vector<cv::String> filePaths;
        
        // cv::glob recursively or directly lists matching files
        cv::glob(folderPattern, filePaths, false);

        for (const auto& filePath : filePaths) {
            // Check for image extensions
            if (filePath.find(".png") == std::string::npos && 
                filePath.find(".jpg") == std::string::npos) {
                continue;
            }

            cv::Mat img;
            if (!Utils::loadImage(filePath, img)) continue;

            // Pipeline: remove noise -> enhance -> predict
            cv::Mat denoised = Preprocessing::removeNoise(img);
            cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
            std::string predictedLabel = classifier.predict(enhanced);

            allGroundTruths.push_back(cat);
            allPredictions.push_back(predictedLabel);

            // Extract pure filename (e.g., "image.png")
            size_t lastSlash = filePath.find_last_of("/\\");
            std::string filename = (lastSlash == std::string::npos) ? filePath : filePath.substr(lastSlash + 1);

            // Save .txt prediction
            Utils::savePredictionTxt(outputDir, filename, predictedLabel);

            // Overlay label on bottom-left and save annotated image
            cv::Mat annotated = img.clone();
            Utils::overlayLabel(annotated, predictedLabel);
            cv::imwrite(outputDir + "/" + filename, annotated);
        }
    }

    // Evaluation summary
    Utils::EvaluationSummary summary = Utils::evaluate(allGroundTruths, allPredictions);
    std::cout << "\n================ EVALUATION SUMMARY ================\n";
    std::cout << "Overall Accuracy: " << summary.overallAccuracy * 100.0 << "%\n";

    return 0;
}