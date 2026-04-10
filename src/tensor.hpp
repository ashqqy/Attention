#pragma once

#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

#include "validation.hpp"

namespace attn {

class Tensor {
  public:
    Tensor(std::size_t n_batches, std::size_t n_rows, std::size_t n_cols)
        : n_batches_(n_batches), n_rows_(n_rows), n_cols_(n_cols),
          data_(n_batches * n_rows * n_cols) {
        details::validate_dimensions(n_batches, n_rows, n_cols);
    }

    template <std::input_iterator InputIter>
    Tensor(std::size_t n_batches, std::size_t n_rows, std::size_t n_cols, InputIter begin,
           InputIter end)
        : n_batches_(n_batches), n_rows_(n_rows), n_cols_(n_cols), data_(begin, end) {
        details::validate_dimensions(n_batches, n_rows, n_cols);
        details::validate_data_size(*this);
    }

  public:
    float& operator[](std::size_t global_idx) noexcept { return data_[global_idx]; }
    const float& operator[](std::size_t global_idx) const noexcept { return data_[global_idx]; }

    float& operator()(std::size_t b, std::size_t i, std::size_t j) noexcept {
        return data_[b * get_rows() * get_cols() + i * get_cols() + j];
    }

    const float& operator()(std::size_t b, std::size_t i, std::size_t j) const noexcept {
        return data_[b * get_rows() * get_cols() + i * get_cols() + j];
    }

    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    std::size_t get_batch() const noexcept { return n_batches_; }
    std::size_t get_rows() const noexcept { return n_rows_; }
    std::size_t get_cols() const noexcept { return n_cols_; }
    std::size_t get_n_elems() const noexcept { return data_.size(); }

  private:
    std::size_t n_batches_ = 0;
    std::size_t n_rows_ = 0;
    std::size_t n_cols_ = 0;

    std::vector<float> data_;
};

} // namespace attn
