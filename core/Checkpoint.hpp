#pragma once

#include "../core/Tensor.hpp"
#include <fstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace aresml {

constexpr uint32_t ARESML_MAGIC = 0x41415245;  // "AREM" little endian
constexpr uint32_t ARESML_VERSION = 1;

struct CheckpointError : std::runtime_error {
    CheckpointError(const std::string& msg) : std::runtime_error(msg) {}
};

struct CheckpointHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_tensors;
    uint8_t compression;
    
    CheckpointHeader() : magic(ARESML_MAGIC), version(ARESML_VERSION), 
                        num_tensors(0), compression(0) {}
};

class Module {
public:
    virtual std::unordered_map<std::string, Tensor*> state_dict() = 0;
    virtual void load_state_dict(const std::unordered_map<std::string, Tensor*>& state) = 0;
    virtual ~Module() = default;
};

class Checkpoint {
public:
    static void save_model(const std::string& path, Module& model) {
        auto state = model.state_dict();
        
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw CheckpointError("Cannot open file for writing: " + path);
        }
        
        CheckpointHeader header;
        header.num_tensors = static_cast<uint32_t>(state.size());
        
        file.write(reinterpret_cast<const char*>(&header.magic), sizeof(header.magic));
        file.write(reinterpret_cast<const char*>(&header.version), sizeof(header.version));
        file.write(reinterpret_cast<const char*>(&header.num_tensors), sizeof(header.num_tensors));
        file.write(reinterpret_cast<const char*>(&header.compression), sizeof(header.compression));
        
        for (auto& pair : state) {
            const std::string& name = pair.first;
            Tensor* tensor = pair.second;
            
            uint32_t name_len = static_cast<uint32_t>(name.size());
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.c_str(), name_len);
            
            uint8_t rank = static_cast<uint8_t>(tensor->shape.n);
            file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
            
            for (size_t i = 0; i < tensor->shape.n; ++i) {
                uint32_t dim = static_cast<uint32_t>(tensor->shape.d[i]);
                file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }
            
            uint32_t size = static_cast<uint32_t>(tensor->shape.size());
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            
            if (tensor->data) {
                const float* data = tensor->data.get() + tensor->offset;
                file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
            }
        }
        
        file.close();
    }
    
    static void load_model(const std::string& path, Module& model) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw CheckpointError("Cannot open file for reading: " + path);
        }
        
        CheckpointHeader header;
        file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
        file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
        file.read(reinterpret_cast<char*>(&header.num_tensors), sizeof(header.num_tensors));
        file.read(reinterpret_cast<char*>(&header.compression), sizeof(header.compression));
        
        if (header.magic != ARESML_MAGIC) {
            throw CheckpointError("Invalid checkpoint magic number");
        }
        
        if (header.version != ARESML_VERSION) {
            throw CheckpointError("Unsupported checkpoint version");
        }
        
        auto state = model.state_dict();
        std::unordered_map<std::string, Tensor*> loaded;
        
        for (uint32_t i = 0; i < header.num_tensors; ++i) {
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            
            std::string name(name_len, ' ');
            file.read(&name[0], name_len);
            
            uint8_t rank;
            file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
            
            Shape shape;
            shape.n = rank;
            for (size_t j = 0; j < rank; ++j) {
                uint32_t dim;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                shape.d[j] = dim;
            }
            
            uint32_t size;
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            
            auto it = state.find(name);
            if (it == state.end()) {
                std::cerr << "[CHECKPOINT] Warning: extra tensor in file: " << name << "\n";
                file.seekg(size * sizeof(float), std::ios::cur);
                continue;
            }
            
            Tensor* tensor = it->second;
            if (tensor->shape.size() != shape.size()) {
                std::ostringstream ss;
                ss << "Shape mismatch for " << name << ": expected " << tensor->shape.size() 
                   << ", got " << shape.size();
                throw CheckpointError(ss.str());
            }
            
            if (tensor->data) {
                float* data = tensor->data.get() + tensor->offset;
                file.read(reinterpret_cast<char*>(data), size * sizeof(float));
            } else {
                file.seekg(size * sizeof(float), std::ios::cur);
            }
            
            loaded[name] = tensor;
        }
        
        model.load_state_dict(loaded);
        file.close();
    }
    
    static uint32_t count_tensors(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return 0;
        }
        
        CheckpointHeader header;
        file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
        file.read(reinterpret_cast<char*>(&header.num_tensors), sizeof(header.num_tensors));
        
        return header.num_tensors;
    }
    
    static void print_info(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open: " << path << "\n";
            return;
        }
        
        CheckpointHeader header;
        file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
        file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
        file.read(reinterpret_cast<char*>(&header.num_tensors), sizeof(header.num_tensors));
        
        std::cout << "=== Checkpoint Info ===\n";
        std::cout << "File: " << path << "\n";
        std::cout << "Magic: " << std::hex << header.magic << std::dec << "\n";
        std::cout << "Version: " << header.version << "\n";
        std::cout << "Tensors: " << header.num_tensors << "\n";
    }
};

class CheckpointInspector {
public:
    static void list_tensors(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw CheckpointError("Cannot open: " + path);
        }
        
        CheckpointHeader header;
        file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
        file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
        file.read(reinterpret_cast<char*>(&header.num_tensors), sizeof(header.num_tensors));
        
        std::cout << "=== Tensors in checkpoint ===\n";
        
        for (uint32_t i = 0; i < header.num_tensors; ++i) {
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            
            std::string name(name_len, ' ');
            file.read(&name[0], name_len);
            
            uint8_t rank;
            file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
            
            std::cout << "  " << i << ": " << name << " (";
            for (size_t j = 0; j < rank; ++j) {
                uint32_t dim;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                if (j > 0) std::cout << "x";
                std::cout << dim;
            }
            std::cout << ")\n";
            
            if (rank < header.num_tensors) {
                file.seekg(-static_cast<int>(rank * sizeof(uint32_t)), std::ios::cur);
            }
            
            uint32_t size;
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            file.seekg(size * sizeof(float), std::ios::cur);
        }
    }
};

}