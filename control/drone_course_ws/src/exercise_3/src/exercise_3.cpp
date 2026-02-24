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
#include <cmath>
#include <limits>
#include <exercise_3/exercise_3.hpp>

namespace drone_course
{

  DroneCourseExercise3::DroneCourseExercise3(
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
        std::bind(&DroneCourseExercise3::state_subscription_callback, this, std::placeholders::_1));

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
        std::bind(&DroneCourseExercise3::timer_callback, this));

    dt_ = 1.0f / timer_freq;

    RCLCPP_INFO(this->get_logger(), "DroneCourseExercise3 initialized\n");
  }

  DroneCourseExercise3::~DroneCourseExercise3() {}

  std::vector<double> DroneCourseExercise3::solve_linear_system(
      std::vector<std::vector<double>> A, std::vector<double> b)
  {
    int N = static_cast<int>(b.size());
    for (int col = 0; col < N; col++)
    {
      int pivot = col;
      for (int row = col + 1; row < N; row++)
        if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
          pivot = row;
      std::swap(A[col], A[pivot]);
      std::swap(b[col], b[pivot]);
      for (int row = col + 1; row < N; row++)
      {
        double f = A[row][col] / A[col][col];
        for (int k = col; k < N; k++)
          A[row][k] -= f * A[col][k];
        b[row] -= f * b[col];
      }
    }
    std::vector<double> x(N);
    for (int i = N - 1; i >= 0; i--)
    {
      x[i] = b[i];
      for (int j = i + 1; j < N; j++)
        x[i] -= A[i][j] * x[j];
      x[i] /= A[i][i];
    }
    return x;
  }

  void DroneCourseExercise3::build_spline()
  {
    int N = static_cast<int>(waypoints_.size());
    spline_M_.assign(N, {0.0, 0.0, 0.0});

    // C2 continuity
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; i++)
    {
      A[i][i] = 4.0;
      A[i][(i + 1) % N] = 1.0;
      A[i][(i - 1 + N) % N] = 1.0;
    }

    for (int k = 0; k < 3; k++)
    {
      std::vector<double> d(N);
      for (int i = 0; i < N; i++)
      {
        d[i] = 6.0 * (waypoints_[(i + 1) % N][k] - 2.0 * waypoints_[i][k] + waypoints_[(i - 1 + N) % N][k]);
      }
      auto M = solve_linear_system(A, d);
      for (int i = 0; i < N; i++)
        spline_M_[i][k] = M[i];
    }
  }

  std::array<double, 3> DroneCourseExercise3::sample_spline(double t) const
  {
    int N = static_cast<int>(waypoints_.size());
    double t_mod = std::fmod(t, static_cast<double>(N));
    if (t_mod < 0.0)
      t_mod += N;

    int i = static_cast<int>(t_mod);
    double s = t_mod - i;
    int j = (i + 1) % N;

    std::array<double, 3> result;
    for (int k = 0; k < 3; k++)
    {
      double p_i = waypoints_[i][k];
      double p_j = waypoints_[j][k];
      double m_i = spline_M_[i][k];
      double m_j = spline_M_[j][k];
      result[k] = (1.0 - s) * p_i + s * p_j + (m_i * (1.0 - s) * ((1.0 - s) * (1.0 - s) - 1.0) + m_j * s * (s * s - 1.0)) / 6.0;
    }
    return result;
  }

  std::array<double, 3> DroneCourseExercise3::sample_spline_deriv(double t) const
  {
    int N = static_cast<int>(waypoints_.size());
    double t_mod = std::fmod(t, static_cast<double>(N));
    if (t_mod < 0.0)
      t_mod += N;

    int i = static_cast<int>(t_mod);
    double s = t_mod - i;
    int j = (i + 1) % N;

    std::array<double, 3> result;
    for (int k = 0; k < 3; k++)
    {
      double p_i = waypoints_[i][k];
      double p_j = waypoints_[j][k];
      double m_i = spline_M_[i][k];
      double m_j = spline_M_[j][k];
      result[k] = (p_j - p_i) + (m_i * (1.0 - 3.0 * (1.0 - s) * (1.0 - s)) + m_j * (3.0 * s * s - 1.0)) / 6.0;
    }
    return result;
  }

  double DroneCourseExercise3::find_closest_t(double x, double y, double z) const
  {
    int N = static_cast<int>(waypoints_.size());

    const int samples_per_seg = 50;
    const double step = 1.0 / samples_per_seg;

    double best_t = traj_t_;
    double best_dist_sq = std::numeric_limits<double>::max();

    for (int s = 0; s <= N * samples_per_seg; s++)
    {
      double t = traj_t_ + s * step;
      auto pt = sample_spline(t);
      double dx = pt[0] - x, dy = pt[1] - y, dz = pt[2] - z;
      double dist_sq = dx * dx + dy * dy + dz * dz;
      if (dist_sq < best_dist_sq)
      {
        best_dist_sq = dist_sq;
        best_t = t;
      }
    }
    return best_t;
  }

  double DroneCourseExercise3::find_lookahead_t(double t_proj, double lookahead) const
  {
    int N = static_cast<int>(waypoints_.size());
    const double dt_arc = 0.02;
    const double max_search = static_cast<double>(N);

    double t = t_proj;
    double arc = 0.0;

    while (arc < lookahead && (t - t_proj) < max_search)
    {
      auto p1 = sample_spline(t);
      auto p2 = sample_spline(t + dt_arc);
      double dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
      double seg_len = std::sqrt(dx * dx + dy * dy + dz * dz);

      if (arc + seg_len >= lookahead)
      {
        t += dt_arc * (lookahead - arc) / seg_len;
        break;
      }
      arc += seg_len;
      t += dt_arc;
    }
    return t;
  }

  void DroneCourseExercise3::timer_callback()
  {
    if (!path_received_)
    {
      path_service_request_ = std::make_shared<drone_course_msgs::srv::RequestPath::Request>();
      rclcpp::Client<drone_course_msgs::srv::RequestPath>::SharedFuture future =
          path_service_client_->async_send_request(path_service_request_).future.share();
      future.wait();
      std::vector<drone_course_msgs::msg::Point> path = future.get()->path;
      if (path.size() > 0)
      {
        RCLCPP_INFO(this->get_logger(), "Path received successfully (%zu waypoints)", path.size());
        waypoints_.clear();
        for (const auto &pt : path)
        {
          double x = static_cast<double>(pt.x) + (pt.x < 7.0f ? 1.0 : -1.0);
          waypoints_.push_back({x,
                                static_cast<double>(pt.y),
                                static_cast<double>(pt.z)});
        }
        build_spline();
        traj_ready_ = true;
        path_received_ = true;
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to get path");
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

      rclcpp::Client<as2_msgs::srv::SetControlMode>::SharedFuture future =
          control_mode_service_client_->async_send_request(control_mode_request).future.share();
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
      }
    }

    if (!traj_ready_ || !control_mode_set_)
      return;

    // Read current drone state
    double current_x = state_pose_.pose.position.x;
    double current_y = state_pose_.pose.position.y;
    double current_z = state_pose_.pose.position.z;

    traj_t_ = find_closest_t(current_x, current_y, current_z);
    double t_look = find_lookahead_t(traj_t_, lookahead_dist_);
    auto lookahead_pt = sample_spline(t_look);

    double dx = lookahead_pt[0] - current_x;
    double dy = lookahead_pt[1] - current_y;
    double dz = lookahead_pt[2] - current_z;

    double Kpp = desired_speed_ / lookahead_dist_;
    double velocity_x = Kpp * dx;
    double velocity_y = Kpp * dy;
    double velocity_z = Kpp * dz;

    double speed = std::sqrt(velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z);
    if (speed > desired_speed_)
    {
      double scale = desired_speed_ / speed;
      velocity_x *= scale;
      velocity_y *= scale;
      velocity_z *= scale;
    }

    // Publish velocity command
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
  }

  void DroneCourseExercise3::state_subscription_callback(
      const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    state_pose_ = *msg;
  }

} // namespace drone_course
