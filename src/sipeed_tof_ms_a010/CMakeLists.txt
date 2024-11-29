cmake_minimum_required(VERSION 3.5)
project(sipeed_tof_ms_a010)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Enable warnings for GCC or Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

# Add the executable and its sources
add_executable(publisher 
  src/main.cc 
  src/cJSON.c 
  src/serial.cc 
  src/frame_handle.cc
)

# Link dependencies to the target
ament_target_dependencies(
  publisher 
  rclcpp 
  std_msgs 
  sensor_msgs 
  OpenCV 
  cv_bridge
)

# Include directories (for custom headers)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(publisher 
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
)

# Install the target
install(TARGETS
  publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Enable testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Uncomment the following line to skip linter checks for copyright
  # set(ament_cmake_copyright_FOUND TRUE)
  # Uncomment the following line to skip cpplint checks (only works in a git repo)
  # set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Finalize package
ament_package()
