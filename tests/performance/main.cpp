#include <cstddef>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "tensor.hpp"
#include "math.hpp"

#include "bench_attention.hpp"
#include "bench_multiply.hpp"

int main() {
    bench::run_multiply_benchmarks();
    bench::run_attention_benchmarks();
}
