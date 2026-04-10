#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "common.hpp"
#include "math.hpp"
#include "ops.hpp"
#include "tensor.hpp"

using namespace attn::math;

class TensorMathTorchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        lhs_data.resize(batch_size * lhs_rows * inner_dim);
        rhs_data.resize(batch_size * inner_dim * rhs_cols);
        RandFill(lhs_data, -100.0f, 100.0f);
        RandFill(rhs_data, -100.0f, 100.0f);
    }

    const torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    const int64_t batch_size = 3;
    const int64_t lhs_rows = 128;
    const int64_t inner_dim = 64;
    const int64_t rhs_cols = 128;

    std::vector<float> lhs_data;
    std::vector<float> rhs_data;

    const float epsilon = 1e-2f;
};

TEST_F(TensorMathTorchTest, MatrixTransposeMatchesTorch) {
    Tensor rhs(batch_size, inner_dim, rhs_cols, rhs_data.begin(), rhs_data.end());
    Tensor rhs_transposed(batch_size, rhs_cols, inner_dim);

    transpose(rhs, rhs_transposed);

    auto t_input = torch::from_blob(rhs.data(), {batch_size, inner_dim, rhs_cols}, options);
    auto t_expected = t_input.transpose(1, 2);
    auto t_actual =
        torch::from_blob(rhs_transposed.data(), {batch_size, rhs_cols, inner_dim}, options);

    EXPECT_TRUE(torch::allclose(t_actual, t_expected, epsilon));
}

TEST_F(TensorMathTorchTest, MatrixMultiplyTrMatchesTorch) {
    Tensor lhs(batch_size, lhs_rows, inner_dim, lhs_data.begin(), lhs_data.end());
    Tensor rhs(batch_size, inner_dim, rhs_cols, rhs_data.begin(), rhs_data.end());
    Tensor result_tensor(batch_size, lhs_rows, rhs_cols);

    Tensor rhs_tr(batch_size, rhs_cols, inner_dim);
    transpose(rhs, rhs_tr);
    multiply_tr(lhs, rhs_tr, result_tensor);

    auto t_lhs = torch::from_blob(lhs.data(), {batch_size, lhs_rows, inner_dim}, options);
    auto t_rhs = torch::from_blob(rhs.data(), {batch_size, inner_dim, rhs_cols}, options);
    auto t_expected = torch::matmul(t_lhs, t_rhs);

    auto t_actual =
        torch::from_blob(result_tensor.data(), {batch_size, lhs_rows, rhs_cols}, options);

    EXPECT_TRUE(torch::allclose(t_actual, t_expected, epsilon));
}

