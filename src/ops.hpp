#pragma once

#include <algorithm>
#include <cmath>

#include "math.hpp"
#include "tensor.hpp"

namespace attn::ops {

namespace details {

inline void softmax_row(const Tensor& input, Tensor& result, std::size_t batch_idx,
                        std::size_t row_idx) {
    std::size_t cols = input.get_cols();

    const float* row_start = &input(batch_idx, row_idx, 0);
    const float* row_end = row_start + cols;

    float max_val = *std::max_element(row_start, row_end);

    float* res_row_start = &result(batch_idx, row_idx, 0);
    float* res_row_end = res_row_start + cols;

    float sum_exp = 0.0f;
    for (std::size_t j = 0; j < cols; ++j) {
        float e_j = std::exp(row_start[j] - max_val);
        res_row_start[j] = e_j;
        sum_exp += e_j;
    }

    std::transform(res_row_start, res_row_end, res_row_start,
                   [sum_exp](float val) { return val / sum_exp; });
}

inline void softmax_batch(const Tensor& input, Tensor& result, std::size_t batch_idx) {
    for (std::size_t i = 0; i < input.get_rows(); ++i) {
        softmax_row(input, result, batch_idx, i);
    }
}

} // namespace details

inline void softmax(const Tensor& input, Tensor& result) {
    details::validate_softmax_dimensions(input, result);

    for (std::size_t b = 0; b < input.get_batch(); ++b) {
        details::softmax_batch(input, result, b);
    }
}

inline Tensor attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
}

} // namespace attn::ops
