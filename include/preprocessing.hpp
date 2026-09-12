// MAIN AUTHOR: Elia De Rossi (member A)

#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>

namespace Preprocessing {

    cv::Mat removeNoise(const cv::Mat& inputImage, int kernelSize = 5);     // smooth out high-frequency water noise

    cv::Mat convertToHSV(const cv::Mat& inputImage);                        // convert BGR to HSV for robust color analysis

    cv::Mat enhanceContrast(const cv::Mat& inputImage);                     // handle uneven lighting

}

#endif