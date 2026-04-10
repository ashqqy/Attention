#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "common.hpp"
#include "math.hpp"
#include "ops.hpp"
#include "tensor.hpp"

using namespace attn::ops;

class TensorOpsTorchTest : public ::testing::Test {
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

    const float epsilon = 1e-6f;
};

TEST_F(TensorOpsTorchTest, SoftmaxMatchesTorch) {
    Tensor input(batch_size, lhs_rows, inner_dim, lhs_data.begin(), lhs_data.end());
    Tensor result(batch_size, lhs_rows, inner_dim);

    softmax(input, result);

    auto t_input = torch::from_blob(input.data(), {batch_size, lhs_rows, inner_dim}, options);
    auto t_expected = torch::softmax(t_input, -1);
    auto t_actual = torch::from_blob(result.data(), {batch_size, lhs_rows, inner_dim}, options);

    EXPECT_TRUE(torch::allclose(t_actual, t_expected, epsilon));
}

// auto sums = t_actual.sum(-1);
// EXPECT_TRUE(torch::allclose(sums, torch::ones_like(sums), 1e-6f));
