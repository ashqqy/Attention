#pragma once

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

#include "common.hpp"
#include "tensor.hpp"
#include "math.hpp"

using namespace attn::math;

using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class TensorMathEigenTest : public ::testing::Test {
  protected:
    void SetUp() override {
        lhs_data.resize(M * K);
        rhs_data.resize(K * N);

        RandFill(lhs_data, -1000.0f, 1000.0f);
        RandFill(rhs_data, -1000.0f, 1000.0f);
    }

    const std::size_t M = 12288;
    const std::size_t K = 128;
    const std::size_t N = 12288;

    std::vector<float> lhs_data;
    std::vector<float> rhs_data;

    const float epsilon = 1e-4f;
};

TEST_F(TensorMathEigenTest, MatrixTransposeMatchesEigen) {
    Tensor rhs(1, K, N, rhs_data.begin(), rhs_data.end());
    Tensor rhs_transposed(1, N, K);

    transpose(rhs, rhs_transposed, 0);

    Eigen::Map<const EigenMatrix> eigen_rhs(&rhs[0], K, N);
    EigenMatrix eigen_transposed = eigen_rhs.transpose();

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < K; ++j) {
            EXPECT_NEAR(rhs_transposed(0, i, j), eigen_transposed(i, j), epsilon);
        }
    }
}

TEST_F(TensorMathEigenTest, MatrixMultiplyTrMatchesEigen) {
    Tensor lhs(1, M, K, lhs_data.begin(), lhs_data.end());
    Tensor rhs(1, K, N, rhs_data.begin(), rhs_data.end());
    
    Tensor rhs_transposed(1, N, K);
    Tensor result_tensor(1, M, N);

    transpose(rhs, rhs_transposed, 0);
    multiply_tr(lhs, rhs_transposed, result_tensor, 0);

    Eigen::Map<const EigenMatrix> eigen_lhs(lhs_data.data(), M, K);
    Eigen::Map<const EigenMatrix> eigen_rhs(rhs_data.data(), K, N);
    
    EigenMatrix eigen_result = eigen_lhs * eigen_rhs;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            EXPECT_NEAR(result_tensor(0, i, j), eigen_result(i, j), epsilon);
        }
    }
}
