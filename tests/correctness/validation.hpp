#pragma once

#include <gtest/gtest.h>

#include "tensor.hpp"
#include "tensor_math.hpp"

using namespace attention::tensor;

TEST(TensorValidationTest, InvalidCreationThrows) {
    EXPECT_THROW(Tensor(0, 10, 10), std::invalid_argument);
}

TEST(TensorValidationTest, IteratorConstructorSizeMismatchThrows) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    EXPECT_THROW(Tensor(0, 1, 5, data.begin(), data.end()), std::invalid_argument);
    EXPECT_THROW(Tensor(1, 2, 3, data.begin(), data.end()), std::invalid_argument);
    EXPECT_THROW(Tensor(1, 2, 2, data.begin(), data.end()), std::invalid_argument);
}

TEST(TensorValidationTest, MultiplyDimensionMismatchThrows) {
    Tensor A(1, 10, 20);
    Tensor B(1, 30, 10);
    Tensor C(1, 10, 10);

    EXPECT_THROW(multiply_matrix_naive(A, B, C, 0), std::invalid_argument);
    EXPECT_THROW(multiply_matrix(A, B, C, 0), std::invalid_argument);
    EXPECT_THROW(multiply_matrix_transposed(A, B, C, 0), std::invalid_argument);
}

TEST(TensorValidationTest, MultiplyBatchOutOfBoundsThrows) {
    Tensor A(1, 10, 10);
    Tensor B(1, 10, 10);
    Tensor C(1, 10, 10);

    EXPECT_THROW(multiply_matrix_naive(A, B, C, 5), std::out_of_range);
    EXPECT_THROW(multiply_matrix(A, B, C, 5), std::out_of_range);
    EXPECT_THROW(multiply_matrix_transposed(A, B, C, 5), std::out_of_range);
}

TEST(TensorValidationTest, TransposeBatchOutOfBoundsThrows) {
    Tensor input(2, 10, 20);
    Tensor result(2, 20, 10);

    EXPECT_THROW(transpose_matrix_naive(input, result, 2), std::out_of_range);
    EXPECT_THROW(transpose_matrix(input, result, 5), std::out_of_range);
}

TEST(TensorValidationTest, TransposeDimensionMismatchThrows) {
    Tensor input(1, 10, 20);

    Tensor result_wrong_1(1, 10, 20);
    Tensor result_wrong_2(1, 20, 20);

    EXPECT_THROW(transpose_matrix_naive(input, result_wrong_1, 0), std::invalid_argument);
    EXPECT_THROW(transpose_matrix(input, result_wrong_2, 0), std::invalid_argument);
}
