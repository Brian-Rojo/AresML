#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "../ir_v2/IRGraphV2.hpp"
#include "CachePlanner.hpp"

namespace aresml {

struct MemoryBuffer {
    std::string name;
    void* data;
    size_t offset;
    size_t size;
    size_t alignment;
    bool is_fused;
    int64_t lifetime_start;
    int64_t lifetime_end;
    
    MemoryBuffer() 
        : data(nullptr), offset(0), size(0), alignment(64)
        , is_fused(false), lifetime_start(-1), lifetime_end(-1) {}
    
    MemoryBuffer(const std::string& n, size_t sz, size_t align = 64)
        : name(n), data(nullptr), offset(0), size(sz), alignment(align)
        , is_fused(false), lifetime_start(-1), lifetime_end(-1) {}
    
    bool overlaps(const MemoryBuffer& other) const {
        if (!data || !other.data) return false;
        size_t end = offset + size;
        size_t other_end = other.offset + other.size;
        return (offset < other_end && other.offset < end);
    }
};

struct MemoryPlan {
    std::vector<MemoryBuffer> buffers;
    std::vector<CacheBlock> blocks;
    size_t total_memory;
    size_t peak_memory;
    float reuse_ratio;
    bool use_fused_buffers;
    
    MemoryPlan() 
        : total_memory(0), peak_memory(0), reuse_ratio(0.0f)
        , use_fused_buffers(true) {}
    
    void add_buffer(const MemoryBuffer& buf) {
        buffers.push_back(buf);
        total_memory += buf.size;
    }
    
    void compute_lifetime(IRGraphV2& graph) {
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto* node = graph.nodes[i].get();
            for (auto* res : node->results) {
                if (res) {
                    for (auto& buf : buffers) {
                        if (buf.name == res->name) {
                            buf.lifetime_start = static_cast<int64_t>(i);
                            buf.lifetime_end = static_cast<int64_t>(graph.nodes.size());
                        }
                    }
                }
            }
        }
    }
    
    size_t compute_peak() {
        peak_memory = 0;
        size_t current = 0;
        
        for (size_t i = 0; i < buffers.size(); ++i) {
            current += buffers[i].size;
            peak_memory = std::max(peak_memory, current);
        }
        
        return peak_memory;
    }
    
    std::string to_string() const {
        std::string s = "MemoryPlan:\n";
        s += "  Total memory: " + std::to_string(total_memory) + " bytes\n";
        s += "  Peak memory: " + std::to_string(peak_memory) + " bytes\n";
        s += "  Buffers: " + std::to_string(buffers.size()) + "\n";
        s += "  Blocks: " + std::to_string(blocks.size()) + "\n";
        s += "  Reuse ratio: " + std::to_string(reuse_ratio) + "\n";
        return s;
    }
};

class MemoryPlanner {
public:
    static MemoryPlan create_plan(IRGraphV2& graph, const CacheBlock& block) {
        MemoryPlan plan;
        
        for (auto& node : graph.nodes) {
            size_t node_size = node->shape.numel() * sizeof(float);
            
            MemoryBuffer buf(node->name, node_size, 64);
            plan.add_buffer(buf);
        }
        
        plan.blocks = CachePlanner::generate_blocking_plan(
            block.block_m, block.block_n, block.block_k
        );
        
        plan.compute_lifetime(graph);
        plan.compute_peak();
        
        plan.reuse_ratio = compute_reuse_ratio(graph, plan);
        
        return plan;
    }
    
    static MemoryPlan create_fused_plan(IRGraphV2& graph, const CacheBlock& block) {
        MemoryPlan plan;
        plan.use_fused_buffers = true;
        
        for (auto& node : graph.nodes) {
            if (node->is_fused()) {
                size_t node_size = node->shape.numel() * sizeof(float);
                MemoryBuffer buf(node->name, node_size, 64);
                buf.is_fused = true;
                plan.add_buffer(buf);
            }
        }
        
        plan.blocks = CachePlanner::generate_blocking_plan(
            block.block_m, block.block_n, block.block_k
        );
        
        plan.compute_peak();
        plan.reuse_ratio = 1.5f;
        
        return plan;
    }
    
    static size_t estimate_savings(IRGraphV2& original, IRGraphV2& fused) {
        size_t orig_size = 0;
        size_t fused_size = 0;
        
        for (auto& node : original.nodes) {
            orig_size += node->shape.numel() * sizeof(float);
        }
        
        for (auto& node : fused.nodes) {
            fused_size += node->shape.numel() * sizeof(float);
        }
        
        return orig_size > fused_size ? orig_size - fused_size : 0;
    }
    
private:
    static float compute_reuse_ratio(IRGraphV2& graph, const MemoryPlan& plan) {
        if (plan.buffers.empty()) return 0.0f;
        
        float total_reuse = 0.0f;
        size_t reused_count = 0;
        
        for (size_t i = 0; i < plan.buffers.size(); ++i) {
            for (size_t j = i + 1; j < plan.buffers.size(); ++j) {
                if (plan.buffers[i].lifetime_start < plan.buffers[j].lifetime_end &&
                    plan.buffers[j].lifetime_start < plan.buffers[i].lifetime_end) {
                    total_reuse += 1.0f;
                    reused_count++;
                }
            }
        }
        
        if (reused_count == 0) return 0.0f;
        return total_reuse / static_cast<float>(reused_count);
    }
};

}
