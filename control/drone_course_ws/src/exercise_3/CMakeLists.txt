# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.5)

# Set the project name
set(PROJECT_NAME exercise_3)
project(${PROJECT_NAME})

# Default to C++17 if not set
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Set Release as default build type if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find required dependencies
set(PROJECT_DEPENDENCIES
  ament_cmake
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  as2_msgs
  drone_course_msgs
)


foreach(DEPENDENCY ${PROJECT_DEPENDENCIES})
  find_package(${DEPENDENCY} REQUIRED)
endforeach()

# Include necessary directories
include_directories(
  include
  include/${PROJECT_NAME}
)

# Set source files
set(SOURCE_CPP_FILES
  src/${PROJECT_NAME}.cpp
)

# # Comment the following block if not export an executable
# Create the node executable
add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp ${SOURCE_CPP_FILES})
ament_target_dependencies(${PROJECT_NAME}_node ${PROJECT_DEPENDENCIES})

# # End of block

# # Comment the following block if not export a library
# Create the shared library
add_library(${PROJECT_NAME}_lib SHARED ${SOURCE_CPP_FILES})
ament_target_dependencies(${PROJECT_NAME}_lib ${PROJECT_DEPENDENCIES})

# # End of block

# Set the public include directories for the library
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install the headers
install(
  DIRECTORY include/
  DESTINATION include
)

# # Comment the following block if not export an executable
# Install the node executable
install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# # End of block

# # Comment the following block if not export a library
# Install the shared library
install(
  TARGETS ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export the libraries
ament_export_libraries(
  ${PROJECT_NAME}_lib
)

# # End of block

# Export the targets
ament_export_targets(
  export_${PROJECT_NAME}
)

# Export the include directories
ament_export_include_directories(
  include
)

# Install the launch directory
# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}
# )

# install(DIRECTORY
#   config
#   DESTINATION share/${PROJECT_NAME}
# )

# # Build tests if testing is enabled
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Package the project using ament
ament_package()
