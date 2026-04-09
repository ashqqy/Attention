#pragma once

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

#include "common.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

using namespace attention::tensor;

using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class TensorMathEigenTest : public ::testing::Test {
  protected:
    void SetUp() override {
        lhs_data.resize(M * K);
        rhs_data.resize(K * N);

        RandFill(lhs_data, -1000.0f, 1000.0f);
        RandFill(rhs_data, -1000.0f, 1000.0f);
    }

    const std::size_t M = 64;
    const std::size_t K = 128;
    const std::size_t N = 32;

    std::vector<float> lhs_data;
    std::vector<float> rhs_data;

    const float epsilon = 1e-4f;
};

TEST_F(TensorMathEigenTest, TransposeMatchesEigen) {
    Tensor rhs(1, K, N, rhs_data.begin(), rhs_data.end());
    Tensor rhs_transposed(1, N, K);

    transpose_matrix(rhs, rhs_transposed, 0);

    Eigen::Map<const EigenMatrix> eigen_rhs(&rhs[0], K, N);
    EigenMatrix eigen_transposed = eigen_rhs.transpose();

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < K; ++j) {
            EXPECT_NEAR(rhs_transposed(0, i, j), eigen_transposed(i, j), epsilon);
        }
    }
}

TEST_F(TensorMathEigenTest, MultiplyTransposedMatchesEigen) {
    Tensor lhs(1, M, K, lhs_data.begin(), lhs_data.end());
    Tensor rhs(1, K, N, rhs_data.begin(), rhs_data.end());
    
    Tensor rhs_transposed(1, N, K);
    Tensor result_tensor(1, M, N);

    transpose_matrix(rhs, rhs_transposed, 0);
    multiply_matrix(lhs, rhs_transposed, result_tensor, 0);

    Eigen::Map<const EigenMatrix> eigen_lhs(&lhs[0], M, K);
    Eigen::Map<const EigenMatrix> eigen_rhs(&rhs[0], K, N);
    
    EigenMatrix eigen_result = eigen_lhs * eigen_rhs;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            EXPECT_NEAR(result_tensor(0, i, j), eigen_result(i, j), epsilon);
        }
    }
}
