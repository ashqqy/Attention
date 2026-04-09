#pragma once

#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

#include "tensor_validate.hpp"

namespace attention::tensor {

class Tensor {
  public: // constructors
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

  public: // observers
    float& operator[](std::size_t global_idx) noexcept { return data_[global_idx]; }
    const float& operator[](std::size_t global_idx) const noexcept { return data_[global_idx]; }

    float& operator()(std::size_t b, std::size_t i, std::size_t j) noexcept {
        return data_[b * get_rows() * get_cols() + i * get_cols() + j];
    }

    const float& operator()(std::size_t b, std::size_t i, std::size_t j) const noexcept {
        return data_[b * get_rows() * get_cols() + i * get_cols() + j];
    }

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

std::ostream& operator<<(std::ostream& ostream, const Tensor& tensor) {
    if (tensor.get_batch() > 1) {
        ostream << "Cannot print the tensor with a batch number greater than 1\n";
        return ostream;
    }

    for (std::size_t i = 0; i < tensor.get_rows(); ++i) {
        for (std::size_t j = 0; j < tensor.get_cols(); ++j) {
            ostream << tensor(0, i, j) << " ";
        }
        ostream << "\n";
    }

    return ostream;
}

} // namespace attention::tensor
