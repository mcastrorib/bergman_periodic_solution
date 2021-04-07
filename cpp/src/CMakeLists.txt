cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# set the project name
project(BERGMAN_SOLUTION VERSION 0.1 LANGUAGES C CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# option(USE_PATHFINDER "Use provided Pathfinder internal lib implementation" ON)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(BergmanSolution_config.h.in BergmanSolution_config.h)

# include OpenMP package
find_package( OpenMP REQUIRED )

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# include Eigen package
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(/usr/include/eigen3)


# add the executable
add_executable(bergman main.cpp)

target_link_libraries(bergman PUBLIC ${EXTRA_LIBS} OpenMP::OpenMP_CXX Eigen3::Eigen)


# add the binary tree to the search path for include files
# so that we will find *config.h
target_include_directories(bergman PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
install(TARGETS bergman DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/pathfinder_config.h"
  DESTINATION include
  )
