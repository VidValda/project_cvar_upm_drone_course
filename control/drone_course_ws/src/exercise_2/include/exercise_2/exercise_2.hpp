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

#ifndef EXERCISE_2__EXERCISE_2_HPP_
#define EXERCISE_2__EXERCISE_2_HPP_

#include <string>
#include <chrono>
#include <memory>
#include <array>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "as2_msgs/srv/set_control_mode.hpp"
#include "as2_msgs/msg/control_mode.hpp"
#include "drone_course_msgs/srv/request_path.hpp"

namespace drone_course
{

  /**
   * @brief Class DroneCourseExercise2
   */
  class DroneCourseExercise2 : public rclcpp::Node
  {
  public:
    /**
     * @brief Construct a new DroneCourse object
     *
     * @param node_name Node name
     * @param options Node options
     */
    explicit DroneCourseExercise2(
        const std::string &node_name = "drone_course_exercise_2_node",
        const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    /**
     * @brief Destroy the DroneCourseExercise2 object
     */
    ~DroneCourseExercise2();

  private:
    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vel_pub_;

    // Timers
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::CallbackGroup::SharedPtr callback_group_;

    // Service clients
    rclcpp::Client<drone_course_msgs::srv::RequestPath>::SharedPtr path_service_client_;
    rclcpp::Client<as2_msgs::srv::SetControlMode>::SharedPtr control_mode_service_client_;
    drone_course_msgs::srv::RequestPath::Request::SharedPtr path_service_request_;
    drone_course_msgs::srv::RequestPath::Response::SharedPtr path_service_response_;

    // Class variables
    geometry_msgs::msg::PoseStamped state_pose_;
    bool control_mode_set_ = false;
    bool path_received_ = false;
    int path_index_ = 0;
    double dt_ = 0.01;

    std::array<float, 12> path_ = {};

    // PID state
    double integral_x_ = 0.0;
    double integral_y_ = 0.0;
    double integral_z_ = 0.0;
    double previous_error_x_ = 0.0;
    double previous_error_y_ = 0.0;
    double previous_error_z_ = 0.0;
    bool pid_initialized_ = false;

  private:
    // Callbacks Subscribers

    /**
     * @brief Subscription callback
     *
     * @param msg geometry_msgs::msg::PoseStamped::SharedPtr Message received
     */
    void state_subscription_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    // Callbacks Timers

    /**
     * @brief Timer callback
     */
    void timer_callback();
  };
} // namespace drone_course

#endif // EXERCISE_2__EXERCISE_2_HPP_
