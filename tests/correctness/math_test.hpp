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
        bench::RandFill(lhs_data, -1.0f, 1.0f);
        bench::RandFill(rhs_data, -1.0f, 1.0f);
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

TEST_F(TensorMathTorchTest, MatrixMultiplyMatchesTorch) {
    attn::Tensor lhs = attn::Tensor::make_view(lhs_data.data(), batch_size, lhs_rows, inner_dim);
    attn::Tensor rhs = attn::Tensor::make_view(rhs_data.data(), batch_size, inner_dim, rhs_cols);
    attn::Tensor rhs_tr = attn::math::transpose(rhs);

    attn::Tensor result_naive = attn::math::multiply_naive(lhs, rhs);
    attn::Tensor result_tr = attn::math::multiply_tr(lhs, rhs_tr);
    attn::Tensor result_cf = attn::math::multiply_cf(lhs, rhs);

    auto t_lhs = torch::from_blob(lhs.data(), {batch_size, lhs_rows, inner_dim}, options);
    auto t_rhs = torch::from_blob(rhs.data(), {batch_size, inner_dim, rhs_cols}, options);
    auto t_expected = torch::matmul(t_lhs, t_rhs);

    auto t_actual_naive = torch::from_blob(result_naive.data(), {batch_size, lhs_rows, rhs_cols}, options);
    auto t_actual_tr = torch::from_blob(result_tr.data(), {batch_size, lhs_rows, rhs_cols}, options);
    auto t_actual_cf = torch::from_blob(result_cf.data(), {batch_size, lhs_rows, rhs_cols}, options);

    EXPECT_TRUE(torch::allclose(t_actual_naive, t_expected, epsilon, epsilon));
    EXPECT_TRUE(torch::allclose(t_actual_tr, t_expected, epsilon, epsilon));
    EXPECT_TRUE(torch::allclose(t_actual_cf, t_expected, epsilon, epsilon));
}

