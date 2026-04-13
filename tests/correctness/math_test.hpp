#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "common.hpp"
#include "math.hpp"
#include "ops.hpp"
#include "tensor.hpp"

class TensorMathTorchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        lhs_data.resize(batch_size * lhs_rows * inner_dim);
        rhs_data.resize(batch_size * inner_dim * rhs_cols);
        RandFill(lhs_data, -1.0f, 1.0f);
        RandFill(rhs_data, -1.0f, 1.0f);
    }

    const torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    const int64_t batch_size = 3;
    const int64_t lhs_rows = 128;
    const int64_t inner_dim = 64;
    const int64_t rhs_cols = 128;

    std::vector<float> lhs_data;
    std::vector<float> rhs_data;

    const float epsilon = 1e-3f;
};

TEST_F(TensorMathTorchTest, MatrixTransposeMatchesTorch) {
    attn::Tensor rhs = attn::Tensor::make_view(rhs_data.data(), batch_size, inner_dim, rhs_cols);
    attn::Tensor rhs_tr = attn::math::transpose(rhs);

    auto t_input = torch::from_blob(rhs_data.data(), {batch_size, inner_dim, rhs_cols}, options);
    auto t_actual = torch::from_blob(rhs_tr.data(), {batch_size, rhs_cols, inner_dim}, options);
    auto t_expected = t_input.transpose(1, 2);

    EXPECT_TRUE(torch::allclose(t_actual, t_expected, epsilon, epsilon));
}

TEST_F(TensorMathTorchTest, MatrixMultiplyTrMatchesTorch) {
    attn::Tensor lhs = attn::Tensor::make_view(lhs_data.data(), batch_size, lhs_rows, inner_dim);
    attn::Tensor rhs = attn::Tensor::make_view(rhs_data.data(), batch_size, inner_dim, rhs_cols);

    attn::Tensor rhs_tr = attn::math::transpose(rhs);
    attn::Tensor result = attn::math::multiply_tr(lhs, rhs_tr);

    auto t_lhs = torch::from_blob(lhs.data(), {batch_size, lhs_rows, inner_dim}, options);
    auto t_rhs = torch::from_blob(rhs.data(), {batch_size, inner_dim, rhs_cols}, options);
    auto t_expected = torch::matmul(t_lhs, t_rhs);

    auto t_actual = torch::from_blob(result.data(), {batch_size, lhs_rows, rhs_cols}, options);

    float max_diff = torch::abs(t_actual - t_expected).max().item<float>();
    std::cout << "[ MULTIPLY DIFF ] Max absolute error: " << max_diff << std::endl;

    EXPECT_TRUE(torch::allclose(t_actual, t_expected, epsilon, epsilon));
}
