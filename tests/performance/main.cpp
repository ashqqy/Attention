#include <cstddef>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

using namespace attention::tensor;

int main() {
    int batches = 7;
    int rows_a = 10000;
    int cols_a = 7000;
    int cols_b = 5000;

    std::size_t size_a = batches * rows_a * cols_a;
    std::size_t size_b = batches * cols_a * cols_b;

    std::vector<float> v1(size_a);
    std::vector<float> v2(size_b);

    RandFill(v1, -1000.0f, 1000.0f);
    RandFill(v2, -1000.0f, 1000.0f);

    Tensor t1(batches, rows_a, cols_a, v1.begin(), v1.end());
    Tensor t2(batches, cols_a, cols_b, v2.begin(), v2.end());
    Tensor t3(batches, rows_a, cols_b);

    // Tensor t1(batches, rows_a, cols_a, v1.begin(), v1.end());
    // Tensor t2(batches, cols_a, rows_a);


    std::size_t warmup = 2;
    std::size_t repeats = 10;

    std::cout << "\nStarting benchmark for " << rows_a << "x" << cols_a << " * " << cols_a << "x"
              << cols_b << "..." << std::endl;

    double naive_time = profile_function(repeats, warmup, multiply_matrix_naive, t1, t2, t3, 0);
    double time_2d = profile_function(repeats, warmup, multiply_matrix, t1, t2, t3, 0);
    double time_2d_t = profile_function(repeats, warmup, multiply_matrix_transposed, t1, t2, t3, 0);

    // double naive_transpose_time = profile_function(repeats, warmup, Tensor::transpose_batch_naive, t1, t2, 5);
    // double transpose_time = profile_function(repeats, warmup, Tensor::transpose_batch, t1, t2, 5);

    std::cout << "Naive 2d time:  " << naive_time << " ms / iter\n";
    std::cout << "2d time:  " << time_2d << " ms / iter\n";
    std::cout << "2d transposed time:  " << time_2d_t << " ms / iter\n";

    // std::cout << "naive time:  " << naive_transpose_time << " ms / iter\n";
    // std::cout << "time:  " << transpose_time << " ms / iter\n";
}
