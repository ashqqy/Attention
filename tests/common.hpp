#pragma once

#include <concepts>
#include <random>
#include <vector>
#include <functional>
#include <chrono>

namespace bench {

template <std::floating_point T = float>
T RandValue(T min_val, T max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<T> dist(min_val, max_val);
    return dist(gen);
}

template <std::floating_point T = float>
void RandFill(std::vector<T>& vec, T min_val, T max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (auto& elem : vec) {
        elem = dist(gen);
    }
}

template <typename F, typename... Args>
double profile_function(std::size_t repeats, std::size_t warmup_repeats, F&& func, Args&&... args) {
    for (std::size_t i = 0; i < warmup_repeats; ++i) {
        std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    }

    double total_time = 0.0;

    for (std::size_t i = 0; i < repeats; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time += elapsed.count();
    }

    return total_time / repeats;
}

} // namespace bench
