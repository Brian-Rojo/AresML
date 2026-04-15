#pragma once

#include "../../core/Tensor.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace aresml {
namespace engine {
namespace gpt {

struct Sampling {
    static int sample_argmax(const Tensor& logits) {
        size_t vocab_size = logits.shape.size();
        float max_val = -1e9f;
        int max_idx = 0;
        
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits.data[logits.offset + i] > max_val) {
                max_val = logits.data[logits.offset + i];
                max_idx = static_cast<int>(i);
            }
        }
        
        return max_idx;
    }
    
    static int sample_temperature(const Tensor& logits, float temperature = 1.0f) {
        size_t vocab_size = logits.shape.size();
        
        float max_val = -1e9f;
        for (size_t i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, logits.data[logits.offset + i]);
        }
        
        float sum = 0.0f;
        std::vector<float> probs(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            float val = (logits.data[logits.offset + i] - max_val) / temperature;
            probs[i] = std::exp(val);
            sum += probs[i];
        }
        
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float cumulative = 0.0f;
        
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] /= sum;
            cumulative += probs[i];
            if (r <= cumulative) {
                return static_cast<int>(i);
            }
        }
        
        return static_cast<int>(vocab_size - 1);
    }
    
    static int sample_top_k(const Tensor& logits, int top_k = 10) {
        size_t vocab_size = logits.shape.size();
        
        std::vector<std::pair<float, int>> pairs;
        for (size_t i = 0; i < vocab_size; ++i) {
            pairs.push_back({logits.data[logits.offset + i], static_cast<int>(i)});
        }
        
        std::partial_sort(pairs.begin(), pairs.begin() + std::min(static_cast<size_t>(top_k), vocab_size),
                         pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float max_val = -1e9f;
        for (size_t i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, logits.data[logits.offset + i]);
        }
        
        float sum = 0.0f;
        size_t k = std::min(static_cast<size_t>(top_k), vocab_size);
        std::vector<float> probs(k);
        
        for (size_t i = 0; i < k; ++i) {
            float val = (pairs[i].first - max_val);
            probs[i] = std::exp(val);
            sum += probs[i];
        }
        
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float cumulative = 0.0f;
        
        for (size_t i = 0; i < k; ++i) {
            probs[i] /= sum;
            cumulative += probs[i];
            if (r <= cumulative) {
                return pairs[i].second;
            }
        }
        
        return pairs[0].second;
    }
    
    static int sample_top_p(const Tensor& logits, float top_p = 0.9f) {
        size_t vocab_size = logits.shape.size();
        
        float max_val = -1e9f;
        for (size_t i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, logits.data[logits.offset + i]);
        }
        
        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(logits.data[logits.offset + i] - max_val);
            sum += probs[i];
        }
        
        std::vector<std::pair<float, int>> pairs;
        for (size_t i = 0; i < vocab_size; ++i) {
            pairs.push_back({probs[i] / sum, static_cast<int>(i)});
        }
        
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float cumulative = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumulative += pairs[i].first;
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cumulative = 0.0f;
        
        for (size_t i = 0; i < cutoff; ++i) {
            cumulative += pairs[i].first;
            if (r <= cumulative) {
                return pairs[i].second;
            }
        }
        
        return pairs[0].second;
    }
};

}
}
}
