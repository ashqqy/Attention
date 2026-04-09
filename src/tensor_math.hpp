#pragma once

#include <cstddef>
#include <numeric>

#include "tensor.hpp"
#include "tensor_validate.hpp"

namespace attention::tensor {

inline void transpose_matrix_naive(const Tensor& input, Tensor& result, std::size_t batch_idx) {
    details::validate_transpose_dimensions(input, result);
    details::validate_batch_bounds(batch_idx, input, result);

    for (std::size_t i = 0; i < input.get_rows(); ++i) {
        for (std::size_t j = 0; j < input.get_cols(); ++j) {
            result(batch_idx, j, i) = input(batch_idx, i, j);
        }
    }
}

inline void transpose_matrix(const Tensor& input, Tensor& result, std::size_t batch_idx) {
    details::validate_transpose_dimensions(input, result);
    details::validate_batch_bounds(batch_idx, input, result);

    std::size_t input_offset = batch_idx * input.get_cols() * input.get_rows();
    std::size_t result_offset = batch_idx * result.get_cols() * result.get_rows();

    for (std::size_t i = 0; i < input.get_rows(); ++i) {
        std::size_t input_idx = input_offset + i * input.get_cols();
        std::size_t result_idx = result_offset + i;
        for (std::size_t j = 0; j < input.get_cols(); ++j) {
            result[result_idx] = input[input_idx];
            input_idx += 1;
            result_idx += result.get_cols();
        }
    }
}

inline void multiply_matrix_naive(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                                  std::size_t batch_idx) {
    details::validate_multiply_dimensions(lhs, rhs, result);
    details::validate_batch_bounds(batch_idx, lhs, rhs, result);

    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        for (std::size_t j = 0; j < rhs.get_cols(); ++j) {
            float sum = 0;
            for (std::size_t k = 0; k < lhs.get_cols(); ++k) {
                sum += lhs(batch_idx, i, k) * rhs(batch_idx, k, j);
            }
            result(batch_idx, i, j) = sum;
        }
    }
}

inline void multiply_matrix(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                            std::size_t batch_idx) {
    details::validate_multiply_dimensions(lhs, rhs, result);
    details::validate_batch_bounds(batch_idx, lhs, rhs, result);

    std::size_t lhs_offset = batch_idx * lhs.get_cols() * lhs.get_rows();
    std::size_t rhs_offset = batch_idx * rhs.get_cols() * rhs.get_rows();

    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        for (std::size_t j = 0; j < rhs.get_cols(); ++j) {
            float sum = 0;
            std::size_t lhs_idx = lhs_offset + i * lhs.get_cols();
            std::size_t rhs_idx = rhs_offset + j;
            for (std::size_t k = 0; k < lhs.get_cols(); ++k) {
                sum += lhs[lhs_idx] * rhs[rhs_idx];
                lhs_idx += 1;
                rhs_idx += rhs.get_cols();
            }
            result(batch_idx, i, j) = sum;
        }
    }
}

inline void multiply_matrix_transposed(const Tensor& lhs, const Tensor& rhs_transposed,
                                       Tensor& result, std::size_t batch_idx) {
    details::validate_multiply_transposed_dimensions(lhs, rhs_transposed, result);
    details::validate_batch_bounds(batch_idx, lhs, rhs_transposed, result);

    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        for (std::size_t j = 0; j < rhs_transposed.get_rows(); ++j) {
            const float* lhs_row_start = &lhs(batch_idx, i, 0);
            const float* lhs_row_end = lhs_row_start + lhs.get_cols();

            const float* rhs_col_start = &rhs_transposed(batch_idx, j, 0);

            result(batch_idx, i, j) =
                std::inner_product(lhs_row_start, lhs_row_end, rhs_col_start, 0.0f);
        }
    }
}

} // namespace attention::tensor
