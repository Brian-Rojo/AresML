#pragma once

#include "../../nn/transformer/GPTModel.hpp"
#include "../../optim/Adam.hpp"
#include "../../loss/CrossEntropy.hpp"
#include "GPTDataset.hpp"
#include <vector>
#include <iostream>
#include <iomanip>

namespace aresml {
namespace engine {
namespace gpt {

struct GPTTrainer {
    aresml::nn::transformer::GPTModel* model;
    aresml::optim::Adam* optimizer;
    aresml::loss::CrossEntropyLoss loss_fn;
    float learning_rate;
    size_t total_steps;
    float loss;
    
    GPTTrainer(aresml::nn::transformer::GPTModel* m, float lr = 1e-3f)
        : model(m), optimizer(nullptr), learning_rate(lr), total_steps(0), loss(0.0f) {
        
        if (!model) return;
        
        std::vector<Tensor*> params;
        
        params.push_back(&model->token_embedding.weight);
        
        for (auto& block : model->blocks) {
            params.push_back(&block.ln1.weight);
            params.push_back(&block.ln1.bias);
            params.push_back(&block.ln2.weight);
            params.push_back(&block.ln2.bias);
            params.push_back(&block.attn.q_proj.weight);
            params.push_back(&block.attn.k_proj.weight);
            params.push_back(&block.attn.v_proj.weight);
            params.push_back(&block.attn.out_proj.weight);
            params.push_back(&block.mlp.fc1.weight);
            params.push_back(&block.mlp.fc1.bias);
            params.push_back(&block.mlp.fc2.weight);
            params.push_back(&block.mlp.fc2.bias);
        }
        
        params.push_back(&model->final_norm.weight);
        params.push_back(&model->final_norm.bias);
        params.push_back(&model->lm_head->weight);
        
        optimizer = new aresml::optim::Adam(params, lr);
    }
    
    ~GPTTrainer() {
        delete optimizer;
    }
    
    void train_step(const Tensor& input, const Tensor& target) {
        if (!model || !optimizer) return;
        
        get_engine().zero_grad();
        
        Tensor logits = model->forward(input);
        
        Tensor logits_flat = logits.view({logits.shape[0] * logits.shape[1], logits.shape[2]});
        Tensor target_flat = target.view({target.shape[0] * target.shape[1]});
        
        Tensor loss_tensor = loss_fn.forward(logits_flat, target_flat);
        
        loss = 0.0f;
        for (size_t i = 0; i < loss_tensor.shape.size(); ++i) {
            loss += loss_tensor.data[loss_tensor.offset + i];
        }
        
        get_engine().backward(&loss_tensor);
        
        optimizer->step();
        
        total_steps++;
    }
    
    void train_epoch(const GPTDataset& dataset, size_t batch_size = 1) {
        size_t num_batches = dataset.size() / batch_size;
        
        for (size_t i = 0; i < num_batches; ++i) {
            auto [input, target] = dataset.get(i);
            train_step(input, target);
            
            if (total_steps % 10 == 0) {
                std::cout << "Step " << total_steps << " | Loss: " << std::fixed << std::setprecision(4) << loss << std::endl;
            }
        }
    }
    
    void print_stats() {
        std::cout << "=== Training Stats ===" << std::endl;
        std::cout << "Total Steps: " << total_steps << std::endl;
        std::cout << "Last Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
        std::cout << "Learning Rate: " << learning_rate << std::endl;
    }
};

}
}
}
