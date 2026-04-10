#include <gtest/gtest.h>

#ifndef NDEBUG
#include "validation_test.hpp"
#endif

#include "math_test.hpp"
#include "ops_test.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
