#include "MyPositionEstimation.hpp"
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

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

    cv::Mat r, t;
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    std::vector<cv::Mat> a;
    a.push_back(R);
    a.push_back(t);
    estimation_matrix = a;
}

void MyPositionEstimation::pos_estimate_3d3d(
    cv::Mat depth_frame1, cv::Mat depth_frame2, float depth_factor, cv::Mat K,
    std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
    std::vector<cv::DMatch> matches)
{
    std::vector<cv::Point3f> pts1, pts2;
    for (cv::DMatch m : matches)
    {
        auto d1 = depth_frame1.ptr<uint16_t>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        auto d2 = depth_frame2.ptr<uint16_t>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) // bad depth
            continue;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) * depth_factor;
        float dd2 = float(d2) * depth_factor;
        pts1.push_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cv::Point3f p1, p2; // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0)
    {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    std::vector<cv::Mat> a;
    cv::Mat R, t;

    // convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2));
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

    a.push_back(R);
    a.push_back(t);
    estimation_matrix = a;
}

std::vector<cv::Mat> MyPositionEstimation::return_estimation(void)
{
    return estimation_matrix;
}
