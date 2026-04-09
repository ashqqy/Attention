find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(correctness_tests correctness/main.cpp)
add_executable(performance_tests performance/main.cpp)

set(TARGETS 
    correctness_tests 
    performance_tests
)

foreach(TARGET_NAME IN LISTS TARGETS)
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        tensor_lib
    )

    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPT})
    target_link_options(${TARGET_NAME} PRIVATE ${LINK_OPT})
    
endforeach()

target_include_directories(correctness_tests PRIVATE 
    ${GTEST_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(correctness_tests PRIVATE
    ${GTEST_LIBRARIES}
)
