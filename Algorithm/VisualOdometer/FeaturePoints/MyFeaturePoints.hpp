#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

class MyFeaturePoints
{
private:
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> pre_keypoints, now_keypoints;

public:
    MyFeaturePoints();
    ~MyFeaturePoints();

    void orb_match(
        cv::Mat pre_frame, cv::Mat now_frame,
        double min_dist, double max_dist,
        cv::DescriptorMatcher::MatcherType match_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    std::vector<cv::DMatch> return_matches(void);
    std::vector<std::vector<cv::KeyPoint>> return_keypoints(void);
};
