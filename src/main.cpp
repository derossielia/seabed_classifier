#include "preprocessing.hpp"
#include "utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./seabed_classifier <path_to_image>" << std::endl;
        return -1;
    }

    const std::string imagePath = argv[1];
    cv::Mat original;

    if (!Utils::loadImage(imagePath, original)) {
        return -1;
    }

    // Test Preprocessing module
    cv::Mat denoised = Preprocessing::removeNoise(original, 5);
    cv::Mat enhanced = Preprocessing::enhanceContrast(denoised);
    cv::Mat hsv = Preprocessing::convertToHSV(enhanced);

    // Test Utils module: threshold a channel to simulate a segmentation mask
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);

    cv::Mat sampleMask;
    cv::threshold(hsvChannels[1], sampleMask, 60, 255, cv::THRESH_BINARY); // Saturation threshold

    // Generate color overlay
    cv::Mat visualOutput = Utils::blendOverlay(original, sampleMask, cv::Scalar(0, 255, 0), 0.5);

    // Compute metric sanity checks on itself
    double selfIoU = Utils::computeIoU(sampleMask, sampleMask);
    std::cout << "Sanity check - Mask Self-IoU: " << selfIoU << " (Expected: 1.0)" << std::endl;

    // Display results using const string identifiers as per lab guidelines
    const std::string kWinOriginal = "Original Image";
    const std::string kWinEnhanced = "Enhanced Seabed";
    const std::string kWinOverlay  = "Segmentation Overlay";

    cv::namedWindow(kWinOriginal, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(kWinEnhanced, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(kWinOverlay, cv::WINDOW_AUTOSIZE);

    cv::imshow(kWinOriginal, original);
    cv::imshow(kWinEnhanced, enhanced);
    cv::imshow(kWinOverlay, visualOutput);

    cv::waitKey(0);
    return 0;
}