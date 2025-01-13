
cmake_minimum_required(VERSION 3.10)
project(abc_flow)

set(CMAKE_CXX_STANDARD 17)

# VTK support
find_package(VTK REQUIRED)

# TBB support
include(cmake/tbb.cmake)
find_package(TBB REQUIRED)

# OpenVDB support
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Monitor <PackageName>_ROOT variables, as this mechanism is used a lot in OpenVDB.
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()
# Try to use the OpenVDB-provided FindOpenVDB.cmake script.
if (${OPENVDB_ROOT})
    list(PREPEND CMAKE_MODULE_PATH ${OPENVDB_ROOT}/lib/cmake/OpenVDB)
endif()
find_package(OpenVDB 9.0.0 REQUIRED)


add_executable(abc main.cpp ftle.cpp)
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(abc PUBLIC OpenVDB::openvdb TBB::tbb ${VTK_LIBRARIES})