#pragma once

#include <cstddef>
#include <stdexcept>

namespace attn::details {

constexpr inline const char* kErrInvalidDimensions =
    "Invalid tensor dimensions: cannot mix zero and non-zero dimensions.";

constexpr inline const char* kErrInputDataAndDimensionsMismatch =
    "Input data size does not match tensor dimensions.";

inline void validate_dimensions(std::size_t batches, std::size_t rows, std::size_t cols) {
#ifndef NDEBUG
    bool has_zero = (batches == 0 || rows == 0 || cols == 0);
    bool all_zero = (batches == 0 && rows == 0 && cols == 0);

    if (has_zero && !all_zero) { throw std::invalid_argument(kErrInvalidDimensions); }
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

} // namespace attn::details


namespace attn::math::details {

constexpr inline const char* kErrTransposeMismatch =
    "Transpose dimension mismatch: result must be (cols x rows) of input.";
constexpr inline const char* kErrTransposeBatchMismatch =
    "Transpose batch mismatch: input and result must have the same number of batches.";

constexpr inline const char* kErrMultiplyMismatch =
    "Multiply dimension mismatch: lhs cols must equal rhs rows.";
constexpr inline const char* kErrMultiplyTrMismatch =
    "Multiply (transposed) dimension mismatch: lhs cols must equal rhs_tr cols.";
constexpr inline const char* kErrMultiplyBatchMismatch =
    "Multiply batch mismatch: lhs, rhs, and result must have the same number of batches.";
constexpr inline const char* kErrMultiplyResultMismatch =
    "Multiply dimension mismatch: result dimensions are incorrect.";

template <typename T>
void validate_transpose_dimensions(const T& input, const T& result) {
#ifndef NDEBUG
    if (input.get_batch() != result.get_batch()) {
        throw std::invalid_argument(kErrTransposeBatchMismatch);
    }

    if (result.get_rows() != input.get_cols() || result.get_cols() != input.get_rows()) {
        throw std::invalid_argument(kErrTransposeMismatch);
    }
#endif
}

template <typename T>
void validate_multiply_dimensions(const T& lhs, const T& rhs, const T& result) {
#ifndef NDEBUG
    if (lhs.get_batch() != rhs.get_batch() || lhs.get_batch() != result.get_batch()) {
        throw std::invalid_argument(kErrMultiplyBatchMismatch);
    }

    if (lhs.get_cols() != rhs.get_rows()) { throw std::invalid_argument(kErrMultiplyMismatch); }

    if (result.get_rows() != lhs.get_rows() || result.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument(kErrMultiplyResultMismatch);
    }
#endif
}

template <typename T>
void validate_multiply_tr_dimensions(const T& lhs, const T& rhs_tr, const T& result) {
#ifndef NDEBUG
    if (lhs.get_batch() != rhs_tr.get_batch() || lhs.get_batch() != result.get_batch()) {
        throw std::invalid_argument(kErrMultiplyBatchMismatch);
    }

    if (lhs.get_cols() != rhs_tr.get_cols()) {
        throw std::invalid_argument(kErrMultiplyTrMismatch);
    }

    if (result.get_rows() != lhs.get_rows() || result.get_cols() != rhs_tr.get_rows()) {
        throw std::invalid_argument(kErrMultiplyResultMismatch);
    }
#endif
}

namespace attn::ops::details {

constexpr static const char* kErrSoftmaxMismatch =
    "Softmax dimension mismatch: input and result tensors must have "
    "identical dimensions.";

template <typename T>
inline void validate_softmax_dimensions(const T& input, const T& result) {
    if (input.get_batch() != result.get_batch() || input.get_rows() != result.get_rows() ||
        input.get_cols() != result.get_cols()) {
        throw std::invalid_argument(kErrSoftmaxMismatch);
    }
}

} // namespace attn::ops::details

} // namespace attn::math::details
