#pragma once

#include <cstddef>
#include <stdexcept>

namespace attention::tensor::details {

constexpr inline const char* kErrInvalidDimensions =
    "Invalid tensor dimensions: batches is 0, but rows/cols are not.";
constexpr inline const char* kErrInputDataAndDimensionsMismatch =
    "Input data size does not match tensor dimensions.";
constexpr inline const char* kErrTransposeMismatch =
    "Transpose dimension mismatch: result must be (cols x rows) of input.";
constexpr inline const char* kErrBatchOutOfBounds = "Batch index is out of bounds.";
constexpr inline const char* kErrMultiplyMismatch =
    "Multiply dimension mismatch: lhs cols must equal rhs rows.";
constexpr inline const char* kErrMultiplyTransposedMismatch =
    "Multiply transposed mismatch: lhs cols must equal rhs_transposed cols.";
constexpr inline const char* kErrMultiplyResultMismatch =
    "Result tensor dimensions are incorrect for multiplication.";


void validate_dimensions(std::size_t batches, std::size_t rows, std::size_t cols) {
#ifndef NDEBUG
    if (batches == 0 && (rows != 0 || cols != 0)) {
        throw std::invalid_argument(kErrInvalidDimensions);
    }
#endif
}

template <typename T>
void validate_data_size(const T& tensor) {
#ifndef NDEBUG
    if (tensor.get_n_elems() != tensor.get_batch() * tensor.get_rows() * tensor.get_cols()) {
        throw std::invalid_argument(kErrInputDataAndDimensionsMismatch);
    }
#endif
}

template <typename T>
void validate_transpose_dimensions(const T& input, const T& result) {
#ifndef NDEBUG
    if (result.get_rows() != input.get_cols() || result.get_cols() != input.get_rows()) {
        throw std::invalid_argument(kErrTransposeMismatch);
    }
#endif
}

template <typename... T>
void validate_batch_bounds(std::size_t batch_idx, const T&... tensors) {
#ifndef NDEBUG
    if (((batch_idx >= tensors.get_batch()) || ...)) {
        throw std::out_of_range(kErrBatchOutOfBounds);
    }
#endif
}

template <typename T>
void validate_multiply_dimensions(const T& lhs, const T& rhs, const T& result) {
#ifndef NDEBUG
    if (lhs.get_cols() != rhs.get_rows()) { throw std::invalid_argument(kErrMultiplyMismatch); }
    if (result.get_rows() != lhs.get_rows() || result.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument(kErrMultiplyResultMismatch);
    }
#endif
}

template <typename T>
void validate_multiply_transposed_dimensions(const T& lhs, const T& rhs_transposed,
                                             const T& result) {
#ifndef NDEBUG
    if (lhs.get_cols() != rhs_transposed.get_cols()) {
        throw std::invalid_argument(kErrMultiplyTransposedMismatch);
    }
    if (result.get_rows() != lhs.get_rows() || result.get_cols() != rhs_transposed.get_rows()) {
        throw std::invalid_argument(kErrMultiplyResultMismatch);
    }
#endif
}

} // namespace attention::tensor::details
