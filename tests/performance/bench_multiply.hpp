#include <torch/torch.h>

#include "bench.hpp"
#include "math.hpp"
#include "tensor.hpp"

namespace bench {

void run_multiply_benchmarks() {
    bench::Context ctx;
    bench::Runner runner("Multiply (Q * K^T)", ctx);

    bench::SharedTensor q_shared(ctx.batch_size, ctx.seq_len_q, ctx.d_k);
    bench::SharedTensor k_shared(ctx.batch_size, ctx.seq_len_k, ctx.d_k);

    attn::Tensor q = q_shared.as_tensor();
    attn::Tensor k = k_shared.as_tensor();
    attn::Tensor k_t = attn::math::transpose(k);

    runner.run("Naive multiply", [&]() { math::multiply_naive(q, k_t); });
    runner.run("Transposed multiply", [&]() { math::multiply_tr(q, k); });
    runner.run("Cache-friendly multiply", [&]() { math::multiply_cf(q, k_t); });

    auto t_q = q_shared.as_torch();
    auto t_k = k_shared.as_torch();
    auto t_k_t = t_k.transpose(-2, -1);

    runner.run("LibTorch", [&]() { torch::matmul(t_q, t_k_t); });

    runner.print();
}

} // namespace bench
