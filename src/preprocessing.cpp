// MAIN AUTHOR: Elia De Rossi (member A)
// MAIN AUTHOR: Member A

#include "preprocessing.hpp"
#include <iostream>
#include <vector>

namespace Preprocessing {

    cv::Mat removeNoise(const cv::Mat& inputImage, int kernelSize) {                        // Noise reduction
        if (inputImage.empty()) {
            std::cerr << "Error [Preprocessing]: Empty image passed to removeNoise!" << std::endl;
            return cv::Mat();
        }

        // Kernel dimensions for spatial filtering must be odd positive integers
        if (kernelSize % 2 == 0) {
            kernelSize++;
        }

        cv::Mat smoothed;
        cv::GaussianBlur(inputImage, smoothed, cv::Size(kernelSize, kernelSize), 0);
        return smoothed;
    }

    cv::Mat convertToHSV(const cv::Mat& inputImage) {                                       // Color space conversion
        if (inputImage.empty()) {
            std::cerr << "Error [Preprocessing]: Empty image passed to convertToHSV!" << std::endl;
            return cv::Mat();
        }

        cv::Mat hsvImage;
        cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);
        return hsvImage;
    }

    cv::Mat enhanceContrast(const cv::Mat& inputImage) {                                    // Luminance equalization (for contrast enhancing without hue distorsion)
        if (inputImage.empty()) {
            std::cerr << "Error [Preprocessing]: Empty image passed to enhanceContrast!" << std::endl;
            return cv::Mat();
        }

        cv::Mat ycrcb;
        cv::cvtColor(inputImage, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;                  // Split YCrCb into individual single-channel cv::Mat objects
        cv::split(ycrcb, channels);

        cv::equalizeHist(channels[0], channels[0]);     // Equalize histogram only on channel 0 (Y = Intensity)

        cv::Mat result;                                 // Merge channels back together and convert to BGR
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
        return result;
    }

}