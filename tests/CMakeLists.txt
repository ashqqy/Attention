find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

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
        ${GTEST_LIBRARIES}
    )

    target_compile_options(${TARGET_NAME} PUBLIC ${COMPILE_OPT})
    target_link_options(${TARGET_NAME} PUBLIC ${LINK_OPT})
    
endforeach()
