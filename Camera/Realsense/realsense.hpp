#pragma once

#include "librealsense2/rs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class realsense
{
private:
    rs2::pipeline pipe;
    rs2::pipeline_profile pipe_pro;
    rs2::config cfg;
    rs2::frameset frame_set;
    rs2::video_frame color_frame = frame_set.get_color_frame();
    rs2::depth_frame depth_frame = frame_set.get_depth_frame();
    rs2::motion_frame gyro_frame = frame_set.first_or_default(RS2_STREAM_GYRO);
    rs2::motion_frame accel_frame = frame_set.first_or_default(RS2_STREAM_ACCEL);
    float depth_units;
    bool if_depth, if_color, if_gyro, if_accel;
    double fx = 605.926;
    double fy = 605.7;
    double ppx = 317.4;
    double ppy = 246.356;

    int exce;

public:
    realsense();
    explicit realsense(bool enable_depth, bool enable_color,
                       bool enable_gyro, bool enable_accel);
    ~realsense();
    bool catch_frame(void);
    void frame_solve(void);
    bool color_frame_solve(void);
    bool depth_frame_solve(void);
    bool gyro_frame_solve(void);
    bool accel_frame_solve(void);

    cv::Mat return_color_frame(std::string color_type = "BGR");
    cv::Mat return_depth_frame(void);
    rs2_vector return_gyro_frame(void);
    rs2_vector return_accel_frame(void);
};
