add_library(LSA LeastSquareAdjust.cpp)

target_include_directories(LSA INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS LSA DESTINATION lib)
install(FILES LeastSquareAdjust.h DESTINATION include)