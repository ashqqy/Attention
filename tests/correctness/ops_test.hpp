#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "common.hpp"
#include "math.hpp"
#include "ops.hpp"
#include "tensor.hpp"

using namespace attn;
using namespace attn::ops;

class TensorOpsTorchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        q_data.resize(batch_size * seq_len_q * d_k);
        k_data.resize(batch_size * seq_len_k * d_k);
        v_data.resize(batch_size * seq_len_k * d_v);

        bench::RandFill(q_data, -1.0f, 1.0f);
        bench::RandFill(k_data, -1.0f, 1.0f);
        bench::RandFill(v_data, -1.0f, 1.0f);
    }

    const torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    const int64_t batch_size = 3;
    const int64_t seq_len_q = 128;
    const int64_t seq_len_k = 128;

    const int64_t d_k = 64;
    const int64_t d_v = 128;

    std::vector<float> q_data;
    std::vector<float> k_data;
    std::vector<float> v_data;

    const float epsilon = 1e-6f;
};

TEST_F(TensorOpsTorchTest, SoftmaxMatchesTorch) {
    Tensor input = Tensor::make_view(q_data.data(), batch_size, seq_len_q, d_k);

    auto t_input = torch::from_blob(input.data(), {batch_size, seq_len_q, d_k}, options);
    auto t_expected = torch::softmax(t_input, -1);

    softmax(input);

    EXPECT_TRUE(torch::allclose(t_input, t_expected, epsilon, epsilon));

    auto sums = t_input.sum(-1);
    EXPECT_TRUE(torch::allclose(sums, torch::ones_like(sums), epsilon, epsilon))
        << "Softmax probabilities do not sum to 1.0!";
}

TEST_F(TensorOpsTorchTest, AttentionMatchesTorch1) {
    Tensor Q = Tensor::make_view(q_data.data(), batch_size, seq_len_q, d_k);
    Tensor K = Tensor::make_view(k_data.data(), batch_size, seq_len_k, d_k);
    Tensor V = Tensor::make_view(v_data.data(), batch_size, seq_len_k, d_v);

    Tensor actual_result = attention(Q, K, V);

    auto t_Q = torch::from_blob(q_data.data(), {batch_size, seq_len_q, d_k}, options);
    auto t_K = torch::from_blob(k_data.data(), {batch_size, seq_len_k, d_k}, options);
    auto t_V = torch::from_blob(v_data.data(), {batch_size, seq_len_k, d_v}, options);

    auto t_expected = torch::scaled_dot_product_attention(t_Q, t_K, t_V);

    auto t_actual = torch::from_blob(actual_result.data(), {batch_size, seq_len_q, d_v}, options);

    float max_diff = torch::abs(t_actual - t_expected).max().item<float>();
    std::cout << "[ ATTENTION DIFF ] Max absolute error: " << max_diff << std::endl;

    EXPECT_TRUE(torch::allclose(t_actual, t_expected, epsilon, epsilon));
}
