#include "MyPositionEstimation.hpp"
#include <iostream>

MyPositionEstimation::MyPositionEstimation(/* args */)
{
}

MyPositionEstimation::~MyPositionEstimation()
{
}

void MyPositionEstimation::pos_estimate_2d2d(
    std::vector<cv::KeyPoint> keypoints_1,
    std::vector<cv::KeyPoint> keypoints_2,
    std::vector<cv::DMatch> matches,
    cv::Mat K)
{
    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    if ((points1.size() >= 5) && (points2.size() >= 5))
    {
        //-- 计算本质矩阵
        cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, K);

        std::vector<cv::Mat> a;
        cv::Mat R, t;

        //-- 从本质矩阵中恢复旋转和平移信息.
        if (essential_matrix.cols == 3 && essential_matrix.rows == 3)
        {
            cv::recoverPose(essential_matrix, points1, points2, K, R, t);
            a.push_back(R);
            a.push_back(t);
            estimation_matrix = a;
        }
    }
    else
    {
        std::cout << "points < 5 !!! Not Enough information!!!!" << std::endl;
    }
}

void MyPositionEstimation::pos_estimate_3d2d(
    cv::Mat depth_frame, float depth_factor, cv::Mat K,
    std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
    std::vector<cv::DMatch> matches)
{
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : matches)
    {
        auto d = depth_frame.ptr<uint16_t>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d * depth_factor;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
}

std::vector<cv::Mat> MyPositionEstimation::return_estimation(void)
{
    return estimation_matrix;
}
