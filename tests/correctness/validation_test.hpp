#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include "tensor.hpp"
#include "math.hpp"

using namespace attn;

TEST(TensorValidationTest, InvalidCreationThrows) {
    EXPECT_THROW(Tensor(0, 10, 10), std::invalid_argument);
    EXPECT_THROW(Tensor(1, 0, 10), std::invalid_argument);
    
    EXPECT_NO_THROW(Tensor(0, 0, 0));

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_THROW(Tensor(1, 2, 3, data.begin(), data.end()), std::invalid_argument);
    EXPECT_THROW(Tensor(1, 2, 2, data.begin(), data.end()), std::invalid_argument);
    
    EXPECT_NO_THROW(Tensor(1, 1, 5, data.begin(), data.end()));
}

TEST(TensorValidationTest, TransposeDimensionMismatchThrows) {
    Tensor input(2, 10, 20);
    Tensor result_bad_batch(3, 20, 10);
    Tensor result_bad_shape(2, 10, 20);

    EXPECT_THROW(math::transpose(input, result_bad_batch), std::invalid_argument);
    EXPECT_THROW(math::transpose(input, result_bad_shape), std::invalid_argument);
}

TEST(TensorValidationTest, MultiplyTrDimensionMismatchThrows) {
    Tensor A(2, 10, 20); 
    Tensor B_tr(2, 30, 20);
    Tensor C_valid(2, 10, 30);

    Tensor A_bad_batch(3, 10, 20);
    EXPECT_THROW(math::multiply_tr(A_bad_batch, B_tr, C_valid), std::invalid_argument);

    Tensor B_tr_bad_inner(2, 30, 15);
    EXPECT_THROW(math::multiply_tr(A, B_tr_bad_inner, C_valid), std::invalid_argument);

    Tensor C_bad_shape(2, 10, 25);
    EXPECT_THROW(math::multiply_tr(A, B_tr, C_bad_shape), std::invalid_argument);
}
