#pragma once

#include "librealsense2/rs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

enum color_frame_state_e
{
    BGR = 0,
    RGB,
    GRAY
};

class RealSense
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
    bool if_depth, if_color, if_gyro, if_accel;
    float fx = 605.926;
    float fy = 605.7;
    float ppx = 317.4;
    float ppy = 246.356;
    float depth_factor = 0.001;

    color_frame_state_e color_frame_state;

    int exce;

public:
    RealSense();
    explicit RealSense(bool enable_depth, bool enable_color,
                       bool enable_gyro, bool enable_accel);
    ~RealSense();
    bool catch_frame(void);
    void frame_solve(void);
    bool color_frame_solve(void);
    bool depth_frame_solve(void);
    bool gyro_frame_solve(void);
    bool accel_frame_solve(void);

    cv::Mat return_color_frame(std::string color_type = "BGR", cv::Size size = cv::Size(800, 600));
    cv::Mat return_depth_frame(void);
    rs2_vector return_gyro_frame(void);
    rs2_vector return_accel_frame(void);

    color_frame_state_e return_color_type(void);

    cv::Mat return_camera_inside_param(void);

    float return_camera_depth_factor(void);
};
