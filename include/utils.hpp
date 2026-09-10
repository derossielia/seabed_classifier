// MAIN AUTHOR: Elia De Rossi (member A)

#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace Utils {

// Load an image safely; returns false if the file cannot be opened
bool loadImage(const std::string& filepath, cv::Mat& outputImage, int flags = cv::IMREAD_COLOR);

// Compute Intersection over Union (IoU) between a prediction mask and ground-truth mask
double computeIoU(const cv::Mat& predMask, const cv::Mat& gtMask);

// Compute pixel classification accuracy between predicted and ground truth masks
double computePixelAccuracy(const cv::Mat& predMask, const cv::Mat& gtMask);

// Visual helper: overlay a color-coded mask onto the background with transparency
cv::Mat blendOverlay(const cv::Mat& baseImage, const cv::Mat& mask, const cv::Scalar& color, double alpha = 0.4);

} // namespace Utils

#endif // UTILS_HPP