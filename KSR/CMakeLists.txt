cmake_minimum_required(VERSION 3.16)
project(KSR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Common compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find CGAL
find_package(CGAL QUIET COMPONENTS)
if (CGAL_FOUND)
  message(STATUS "CGAL found")
else()
  message(SEND_ERROR "CGAL library is required")
  return()
endif()

# Find Eigen3 (required for CGAL kinetic surface / Eigen types)
find_package(Eigen3 REQUIRED)
include(CGAL_Eigen3_support)

# Export compile commands for editor tooling (e.g., VS Code C/C++ IntelliSense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find all .cpp files in current directory
file(GLOB CPP_FILES "*.cpp")

# Create executable for each .cpp file
foreach(CPP_FILE ${CPP_FILES})
    # Get filename without extension for executable name
    get_filename_component(EXEC_NAME ${CPP_FILE} NAME_WE)
    
    # Add executable
    add_executable(${EXEC_NAME} ${CPP_FILE})
    
    # Link CGAL and Eigen support to the executable
    target_link_libraries(${EXEC_NAME} PRIVATE CGAL::CGAL)
    if(TARGET CGAL::Eigen3_support)
        target_link_libraries(${EXEC_NAME} PRIVATE CGAL::Eigen3_support)
    endif()
    
    # Ensure Eigen include/linking is available for headers like <Eigen/Dense>
    target_link_libraries(${EXEC_NAME} PRIVATE Eigen3::Eigen)
    
    set_target_properties(${EXEC_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    
    message(STATUS "KSR: Created executable ${EXEC_NAME} from ${CPP_FILE}")
endforeach()
