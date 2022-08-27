#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class MyPositionEstimation
{
private:
    std::vector<cv::Mat> estimation_matrix;

public:
    MyPositionEstimation(/* args */);
    ~MyPositionEstimation();

    void pos_estimate_2d2d(
        std::vector<cv::KeyPoint> keypoints_1,
        std::vector<cv::KeyPoint> keypoints_2,
        std::vector<cv::DMatch> matches,
        cv::Mat K);

    void pos_estimate_3d2d(
        cv::Mat depth_frame, float depth_factor, cv::Mat K,
        std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
        std::vector<cv::DMatch> matches);

    void pos_estimate_3d3d(
        cv::Mat depth_frame1, cv::Mat depth_frame2, float depth_factor, cv::Mat K,
        std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
        std::vector<cv::DMatch> matches);

    std::vector<cv::Mat> return_estimation(void);

    inline cv::Point2d pixel2cam(cv::Point2d p, cv::Mat K)
    {
        return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
    };
};
