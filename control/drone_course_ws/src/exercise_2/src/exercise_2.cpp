// Copyright 2026 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <exercise_2/exercise_2.hpp>

namespace drone_course
{

  DroneCourseExercise2::DroneCourseExercise2(
      const std::string &node_name,
      const rclcpp::NodeOptions &options)
      : rclcpp::Node(node_name, options)
  {
    // QoS profiles
    rclcpp::QoS reliable_qos = rclcpp::QoS(10).reliable();
    rclcpp::QoS best_effort_qos = rclcpp::QoS(10).best_effort();

    callback_group_ =
        this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // Suscribers

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/drone0/self_localization/pose", best_effort_qos,
        std::bind(&DroneCourseExercise2::state_subscription_callback, this, std::placeholders::_1));

    // Publishers

    vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        "/drone0/motion_reference/twist", reliable_qos);

    // Services clients

    control_mode_service_client_ = this->create_client<as2_msgs::srv::SetControlMode>(
        "/drone0/controller/set_control_mode", rmw_qos_profile_services_default, callback_group_);

    path_service_client_ = this->create_client<drone_course_msgs::srv::RequestPath>(
        "/request_path", rmw_qos_profile_services_default, callback_group_);

    // Timers
    double timer_freq = 100.0;
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0f / timer_freq),
        std::bind(&DroneCourseExercise2::timer_callback, this));

    dt_ = 1.0f / timer_freq;

    RCLCPP_INFO(this->get_logger(), "DroneCourseExercise2 initialized\n");
  }

  DroneCourseExercise2::~DroneCourseExercise2() {}

  void DroneCourseExercise2::timer_callback()
  {
    if (!path_received_)
    {
      // (TODO) Exercise 1: Send the request and process the path message
      path_service_request_ = std::make_shared<drone_course_msgs::srv::RequestPath::Request>();
      rclcpp::Client<drone_course_msgs::srv::RequestPath>::SharedFuture future =
          path_service_client_->async_send_request(path_service_request_).future.share();
      future.wait();
      std::vector<drone_course_msgs::msg::Point> path = future.get()->path;
      if (path.size() > 0)
      {
        RCLCPP_INFO(this->get_logger(), "Path received successfully");
        // Process the received path here
        for (size_t i = 0; i < path.size(); i++)
        {
          path_[3 * i] = path[i].x;
          path_[3 * i + 1] = path[i].y;
          path_[3 * i + 2] = path[i].z;
        }
        path_received_ = true;
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to get path");
        path_received_ = false;
      }
    }

    // Check if control mode is set, if not, call service
    if (!control_mode_set_)
    {
      RCLCPP_INFO(this->get_logger(), "Calling control mode service...");
      auto control_mode_request = std::make_shared<as2_msgs::srv::SetControlMode::Request>();
      control_mode_request->control_mode.control_mode = as2_msgs::msg::ControlMode::SPEED;
      control_mode_request->control_mode.yaw_mode = as2_msgs::msg::ControlMode::YAW_SPEED;
      control_mode_request->control_mode.reference_frame =
          as2_msgs::msg::ControlMode::LOCAL_ENU_FRAME;

      // TODO(Exercise 1): Send control mode service request
      rclcpp::Client<as2_msgs::srv::SetControlMode>::SharedFuture future =
          control_mode_service_client_->async_send_request(
                                          control_mode_request)
              .future.share();
      future.wait();
      as2_msgs::srv::SetControlMode::Response::SharedPtr response = future.get();
      if (response->success)
      {
        RCLCPP_INFO(this->get_logger(), "Control mode set successfully");
        control_mode_set_ = true;
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to set control mode");
        control_mode_set_ = false;
      }
    }

    // Desired position reference
    std::array<double, 3> position_ref = {path_[3 * path_index_],
                                          path_[3 * path_index_ + 1],
                                          path_[3 * path_index_ + 2]};

    // Read current drone state
    double current_x = state_pose_.pose.position.x;
    double current_y = state_pose_.pose.position.y;
    double current_z = state_pose_.pose.position.z;

    double position_error_x = position_ref[0] - current_x;
    double position_error_y = position_ref[1] - current_y;
    double position_error_z = position_ref[2] - current_z;
    double position_error_norm = std::sqrt(
        position_error_x * position_error_x + position_error_y * position_error_y + position_error_z * position_error_z);

    // Compute velocity commands
    double Kp = 3.5;
    double Ki = 0.1;
    double Kd = 1.0;
    const double max_velocity = 4.0;

    const double max_integral = max_velocity / Ki;
    integral_x_ = std::clamp(integral_x_ + position_error_x * dt_, -max_integral, max_integral);
    integral_y_ = std::clamp(integral_y_ + position_error_y * dt_, -max_integral, max_integral);
    integral_z_ = std::clamp(integral_z_ + position_error_z * dt_, -max_integral, max_integral);

    // Skip derivative on first call to avoid spike from uninitialized previous_error
    double derivative_x = 0;
    double derivative_y = 0;
    double derivative_z = 0;
    if (pid_initialized_)
    {
      derivative_x = (position_error_x - previous_error_x_) / dt_;
      derivative_y = (position_error_y - previous_error_y_) / dt_;
      derivative_z = (position_error_z - previous_error_z_) / dt_;
    }
    else
    {
      pid_initialized_ = true;
    }

    previous_error_x_ = position_error_x;
    previous_error_y_ = position_error_y;
    previous_error_z_ = position_error_z;

    double velocity_x = std::clamp(
        Kp * position_error_x + Ki * integral_x_ + Kd * derivative_x,
        -max_velocity, max_velocity);
    double velocity_y = std::clamp(
        Kp * position_error_y + Ki * integral_y_ + Kd * derivative_y,
        -max_velocity, max_velocity);
    double velocity_z = std::clamp(
        Kp * position_error_z + Ki * integral_z_ + Kd * derivative_z,
        -max_velocity, max_velocity);

    // Generate motion reference command
    geometry_msgs::msg::TwistStamped velocity_msg;
    velocity_msg.header.stamp = this->get_clock()->now();
    velocity_msg.header.frame_id = "earth";
    velocity_msg.twist.linear.x = velocity_x;
    velocity_msg.twist.linear.y = velocity_y;
    velocity_msg.twist.linear.z = velocity_z;
    velocity_msg.twist.angular.x = 0.0;
    velocity_msg.twist.angular.y = 0.0;
    velocity_msg.twist.angular.z = 0.0;

    if (vel_pub_)
    {
      vel_pub_->publish(velocity_msg);
    }

    if (position_error_norm < 0.2)
    {
      RCLCPP_INFO(this->get_logger(), "Drone has reach the target position");
      path_index_ = (path_index_ + 1) % (path_.size() / 3);
      integral_x_ = 0.0;
      integral_y_ = 0.0;
      integral_z_ = 0.0;
    }
  }

  void DroneCourseExercise2::state_subscription_callback(
      const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    state_pose_ = *msg;
  }
} // namespace drone_course
