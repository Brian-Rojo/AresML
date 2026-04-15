#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <string>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "nn/transformer/TokenEmbedding.hpp"
#include "nn/transformer/MultiHeadSelfAttention.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

// Helper function to create cross entropy loss
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, bool reduction_mean = true) {
    loss::CrossEntropyLoss ce(reduction_mean);
    return ce.forward(logits, targets);
}

// ============================================================================
// TEST OBLIGATORIO: Verificar que TODOS los gradientes fluyen correctamente
// ============================================================================

bool check_grad_not_zero(const Tensor& t, const std::string& name, float threshold = 1e-6f) {
    if (!t.grad || !t.grad->data) {
        std::cerr << "[FAIL] " << name << ": grad tensor is null" << std::endl;
        return false;
    }

    float max_grad = 0.0f;
    float mean_grad = 0.0f;
    size_t n = t.grad->shape.size();

    for (size_t i = 0; i < n; ++i) {
        float g = std::abs(t.grad->data[t.grad->offset + i]);
        max_grad = std::max(max_grad, g);
        mean_grad += g;
    }
    mean_grad /= n;

    if (max_grad < threshold) {
        std::cerr << "[FAIL] " << name << ": gradients are zero (max=" << max_grad << ", mean=" << mean_grad << ")" << std::endl;
        return false;
    }

    std::cout << "[PASS] " << name << ": max_grad=" << max_grad << ", mean_grad=" << mean_grad << std::endl;
    return true;
}

// ============================================================================
// TEST 1: Linear layer gradients
// ============================================================================
bool test_linear_gradients() {
    std::cout << "\n=== TEST 1: Linear Layer Gradients ===" << std::endl;

    // Clear previous parameters
    clear_parameters();

    // Create a simple linear layer
    nn::Linear linear(10, 5, true);

    // Create input
    Tensor input({2, 10}, true);
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Forward pass
    Tensor output = linear.forward(input);

    // Create dummy gradient (all ones)
    output.set_requires_grad(true);
    if (!output.grad) {
        output.grad = std::make_shared<Tensor>(output.shape);
    }
    output.grad->fill(1.0f);

    // Backward pass
    backward(output);

    // Check gradients
    bool pass = true;
    pass &= check_grad_not_zero(linear.weight, "linear.weight");
    pass &= check_grad_not_zero(linear.bias, "linear.bias");

    std::cout << "[TEST 1] " << (pass ? "PASSED" : "FAILED") << std::endl;
    return pass;
}

// ============================================================================
// TEST 2: Embedding layer gradients
// ============================================================================
bool test_embedding_gradients() {
    std::cout << "\n=== TEST 2: Embedding Layer Gradients ===" << std::endl;

    clear_parameters();

    // Create embedding layer
    nn::transformer::TokenEmbedding embedding(100, 16);

    // Create input token IDs (requires_grad=false for token IDs)
    Tensor input_ids({2, 4}, false);  // batch=2, seq_len=4
    int* ids = reinterpret_cast<int*>(input_ids.data.get());
    ids[0] = 1; ids[1] = 2; ids[2] = 3; ids[3] = 4;
    ids[4] = 5; ids[5] = 6; ids[6] = 7; ids[7] = 8;

    // Forward pass - embedding attaches backward op and connects weight as input
    Tensor output = embedding.forward(input_ids);

    // Create dummy gradient and attach to output
    if (!output.grad) {
        output.grad = std::make_shared<Tensor>(output.shape);
    }
    output.grad->fill(1.0f);
    output.requires_grad = true;  // Ensure requires_grad is set

    std::cout << "Output shape: " << output.shape.size() << ", requires_grad: " << output.requires_grad << std::endl;
    std::cout << "Output has op: " << (output.op ? "yes" : "no") << std::endl;
    if (output.op) {
        std::cout << "Op inputs: ";
        for (auto* inp : output.op->get_inputs()) {
            std::cout << "shape=" << (inp ? inp->shape.size() : 0) << " rg=" << (inp ? inp->requires_grad : 0) << " ";
        }
        std::cout << std::endl;
    }

    // Backward pass
    backward(output);

    // Check gradients
    bool pass = true;
    pass &= check_grad_not_zero(embedding.weight, "embedding.weight");

    std::cout << "[TEST 2] " << (pass ? "PASSED" : "FAILED") << std::endl;
    return pass;
}

// ============================================================================
// TEST 3: Attention layer gradients
// ============================================================================
bool test_attention_gradients() {
    std::cout << "\n=== TEST 3: Attention Layer Gradients ===" << std::endl;

    clear_parameters();

    size_t embed_dim = 16;
    size_t num_heads = 2;

    // Create attention layer (this creates 4 Linear layers internally)
    nn::transformer::MultiHeadSelfAttention attention(embed_dim, num_heads, false);

    // Create input
    size_t batch_size = 2;
    size_t seq_len = 4;
    Tensor input({batch_size, seq_len, embed_dim}, true);
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Forward pass
    Tensor output = attention.forward(input, true);
    output.set_requires_grad(true);

    // Create dummy gradient
    if (!output.grad) {
        output.grad = std::make_shared<Tensor>(output.shape);
    }
    output.grad->fill(1.0f);

    // Backward pass
    backward(output);

    // Check gradients - attention has 4 Linear layers with weights
    bool pass = true;
    pass &= check_grad_not_zero(attention.q_proj.weight, "attention.q_proj.weight");
    pass &= check_grad_not_zero(attention.k_proj.weight, "attention.k_proj.weight");
    pass &= check_grad_not_zero(attention.v_proj.weight, "attention.v_proj.weight");
    pass &= check_grad_not_zero(attention.out_proj.weight, "attention.out_proj.weight");

    std::cout << "[TEST 3] " << (pass ? "PASSED" : "FAILED") << std::endl;
    return pass;
}

// ============================================================================
// TEST 4: Cross entropy loss with Linear (simulating GPT lm_head)
// ============================================================================
bool test_cross_entropy_gradients() {
    std::cout << "\n=== TEST 4: Cross Entropy + Linear (LM Head) Gradients ===" << std::endl;

    clear_parameters();

    size_t vocab_size = 50;
    size_t embed_dim = 16;
    size_t seq_len = 4;

    // Create lm_head linear
    nn::Linear lm_head("lm_head", embed_dim, vocab_size, false);

    // Create hidden states
    Tensor hidden({1, seq_len, embed_dim}, true);
    for (size_t i = 0; i < hidden.shape.size(); ++i) {
        hidden.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Forward through lm_head
    Tensor hidden_2d = hidden.view({seq_len, embed_dim});
    Tensor logits_2d = lm_head.forward(hidden_2d);
    Tensor logits = logits_2d.view({1, seq_len, vocab_size});

    // Create target tokens
    Tensor targets({1, seq_len}, false);
    int* targets_data = reinterpret_cast<int*>(targets.data.get());
    targets_data[0] = 5;
    targets_data[1] = 10;
    targets_data[2] = 15;
    targets_data[3] = 20;

    // Cross entropy loss
    Tensor loss = cross_entropy_loss(logits, targets, true);

    // Backward pass
    backward(loss);

    // Check gradients
    bool pass = true;
    pass &= check_grad_not_zero(lm_head.weight, "lm_head.weight");

    std::cout << "[TEST 4] " << (pass ? "PASSED" : "FAILED") << std::endl;
    return pass;
}

// ============================================================================
// TEST 5: Full GPT-like model gradients
// ============================================================================
bool test_gpt_like_gradients() {
    std::cout << "\n=== TEST 5: Full GPT-like Model Gradients ===" << std::endl;

    clear_parameters();

    size_t vocab_size = 100;
    size_t embed_dim = 32;
    size_t num_heads = 2;
    size_t seq_len = 8;
    size_t batch_size = 1;

    // Build GPT-like model manually (to avoid GPTModel dependency on other fixed modules)
    
    // 1. Token embedding
    nn::transformer::TokenEmbedding token_emb(vocab_size, embed_dim);
    
    // 2. LM head
    nn::Linear lm_head("lm_head", embed_dim, vocab_size, false);

    // Input tokens
    Tensor input_ids({batch_size, seq_len}, false);
    int* ids = reinterpret_cast<int*>(input_ids.data.get());
    for (size_t i = 0; i < batch_size * seq_len; ++i) {
        ids[i] = static_cast<int>(rand() % vocab_size);
    }

    // Forward: token embedding
    Tensor embeddings = token_emb.forward(input_ids);
    embeddings.set_requires_grad(true);

    // Forward: lm_head
    Tensor emb_2d = embeddings.view({batch_size * seq_len, embed_dim});
    Tensor logits_2d = lm_head.forward(emb_2d);
    Tensor logits = logits_2d.view({batch_size, seq_len, vocab_size});

    // Targets
    Tensor targets({batch_size, seq_len}, false);
    int* targets_data = reinterpret_cast<int*>(targets.data.get());
    for (size_t i = 0; i < batch_size * seq_len; ++i) {
        targets_data[i] = static_cast<int>(rand() % vocab_size);
    }

    // Loss
    Tensor loss = cross_entropy_loss(logits, targets, true);

    // Backward
    backward(loss);

    // Check ALL parameter gradients
    bool pass = true;
    std::cout << "\nParameter count: " << parameter_count() << std::endl;
    
    pass &= check_grad_not_zero(token_emb.weight, "token_embedding.weight");
    pass &= check_grad_not_zero(lm_head.weight, "lm_head.weight");

    std::cout << "[TEST 5] " << (pass ? "PASSED" : "FAILED") << std::endl;
    return pass;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ARESML - AUTOGRAD GRADIENT FLOW TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    // Enable debug mode for detailed output
    set_debug(true);

    int passed = 0;
    int total = 5;

    if (test_linear_gradients()) passed++;
    if (test_embedding_gradients()) passed++;
    if (test_attention_gradients()) passed++;
    if (test_cross_entropy_gradients()) passed++;
    if (test_gpt_like_gradients()) passed++;

    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    if (passed == total) {
        std::cout << "\n*** ALL TESTS PASSED ***" << std::endl;
        std::cout << "Gradient flow is working correctly!" << std::endl;
        return 0;
    } else {
        std::cerr << "\n*** SOME TESTS FAILED ***" << std::endl;
        std::cerr << "Gradient flow has issues!" << std::endl;
        return 1;
    }
}
