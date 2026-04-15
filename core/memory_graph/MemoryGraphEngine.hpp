#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace aresml {

class MemoryGraphEngine;

struct TensorStorage {
    float* data;
    size_t size;
    size_t capacity;
    int32_t ref_count;
    bool is_view;
    uint64_t id;
    
    std::vector<uint64_t> dependent_ids;
    std::vector<uint64_t> view_ids;
    
    TensorStorage() 
        : data(nullptr), size(0), capacity(0), ref_count(0)
        , is_view(false), id(0) {}
    
    TensorStorage(float* d, size_t sz, size_t cap)
        : data(d), size(sz), capacity(cap), ref_count(1)
        , is_view(false), id(0) {}
    
    bool is_shared() const { return ref_count > 1; }
    bool can_free() const { return ref_count == 0 && dependent_ids.empty(); }
    
    void add_ref() { ref_count++; }
    void release() { if (ref_count > 0) ref_count--; }
};

class MemoryGraphEngine {
public:
    static MemoryGraphEngine& get_instance() {
        static MemoryGraphEngine instance;
        return instance;
    }
    
    void inc_ref(TensorStorage* s) {
        if (s) s->add_ref();
    }
    
    void dec_ref(TensorStorage* s) {
        if (s) {
            s->release();
            maybe_free(s);
        }
    }
    
    struct StorageHandle {
        TensorStorage* storage_;
        MemoryGraphEngine* engine_;
        
        StorageHandle() : storage_(nullptr), engine_(nullptr) {}
        
        StorageHandle(TensorStorage* s, MemoryGraphEngine* e)
            : storage_(s), engine_(e) {
            if (storage_) storage_->add_ref();
        }
        
        StorageHandle(const StorageHandle& other)
            : storage_(other.storage_), engine_(other.engine_) {
            if (storage_) storage_->add_ref();
        }
        
        StorageHandle& operator=(const StorageHandle& other) {
            if (this != &other) {
                release();
                storage_ = other.storage_;
                engine_ = other.engine_;
                if (storage_) storage_->add_ref();
            }
            return *this;
        }
        
        ~StorageHandle() { release(); }
        
        float* data() const { return storage_ ? storage_->data : nullptr; }
        size_t size() const { return storage_ ? storage_->size : 0; }
        size_t capacity() const { return storage_ ? storage_->capacity : 0; }
        int32_t ref_count() const { return storage_ ? storage_->ref_count : 0; }
        uint64_t id() const { return storage_ ? storage_->id : 0; }
        
        bool is_valid() const { return storage_ != nullptr && storage_->data != nullptr; }
        
        TensorStorage* get() const { return storage_; }
        
    private:
        void release() {
            if (storage_ && engine_) {
                storage_->release();
                engine_->maybe_free(storage_);
            }
            storage_ = nullptr;
            engine_ = nullptr;
        }
    };
    
    StorageHandle allocate(size_t size, size_t alignment = 64) {
        size_t total_size = align_size(size, alignment);
        float* data = allocate_aligned(total_size);
        
        auto storage = std::make_unique<TensorStorage>(data, size, total_size);
        storage->id = next_storage_id_++;
        
        uint64_t id = storage->id;
        storages_[id] = std::move(storage);
        
        active_bytes_ += total_size;
        peak_bytes_ = std::max(peak_bytes_, active_bytes_);
        
        return StorageHandle(storages_[id].get(), this);
    }
    
    StorageHandle create_view(StorageHandle base, size_t offset, const std::vector<int64_t>& shape) {
        if (!base.is_valid()) return StorageHandle();
        
        auto* base_storage = base.get();
        base_storage->view_ids.push_back(next_storage_id_);
        
        auto view_storage = std::make_unique<TensorStorage>();
        view_storage->data = base_storage->data + offset;
        view_storage->size = compute_size(shape);
        view_storage->capacity = base_storage->capacity - offset;
        view_storage->ref_count = 1;
        view_storage->is_view = true;
        view_storage->id = next_storage_id_++;
        
        view_storage->dependent_ids.push_back(base_storage->id);
        
        uint64_t id = view_storage->id;
        storages_[id] = std::move(view_storage);
        
        return StorageHandle(storages_[id].get(), this);
    }
    
    void register_dependency(uint64_t from_id, uint64_t to_id) {
        auto it = storages_.find(from_id);
        if (it != storages_.end()) {
            it->second->dependent_ids.push_back(to_id);
        }
    }
    
    void maybe_free(TensorStorage* storage) {
        if (!storage || !storage->can_free()) return;
        
        uint64_t id = storage->id;
        
        for (auto* base_storage : get_dependents(storage)) {
            if (base_storage && base_storage->can_free()) {
                free_storage(base_storage);
            }
        }
        
        free_storage(storage);
    }
    
    void collect_garbage() {
        std::vector<uint64_t> to_free;
        
        for (auto& pair : storages_) {
            auto* storage = pair.second.get();
            if (storage->can_free()) {
                to_free.push_back(pair.first);
            }
        }
        
        for (auto id : to_free) {
            free_storage(storages_[id].get());
            storages_.erase(id);
        }
    }
    
    void clear() {
        for (auto& pair : storages_) {
            free_storage(pair.second.get());
        }
        storages_.clear();
        next_storage_id_ = 0;
        active_bytes_ = 0;
    }
    
    size_t active_storage_count() const { return storages_.size(); }
    size_t active_bytes() const { return active_bytes_; }
    size_t peak_bytes() const { return peak_bytes_; }
    
    void reset_peak() { peak_bytes_ = active_bytes_; }
    
    std::string memory_report() const {
        std::ostringstream ss;
        ss << "MemoryGraph Report:\n";
        ss << "  Active storages: " << storages_.size() << "\n";
        ss << "  Active bytes: " << active_bytes_ << "\n";
        ss << "  Peak bytes: " << peak_bytes_ << "\n";
        
        size_t views = 0;
        size_t shared = 0;
        for (auto& pair : storages_) {
            if (pair.second->is_view) views++;
            if (pair.second->is_shared()) shared++;
        }
        
        ss << "  Views: " << views << "\n";
        ss << "  Shared: " << shared << "\n";
        
        return ss.str();
    }
    
private:
    MemoryGraphEngine() 
        : next_storage_id_(0), active_bytes_(0), peak_bytes_(0) {}
    
    ~MemoryGraphEngine() { clear(); }
    
    MemoryGraphEngine(const MemoryGraphEngine&) = delete;
    MemoryGraphEngine& operator=(const MemoryGraphEngine&) = delete;
    
    static size_t align_size(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    static float* allocate_aligned(size_t size) {
        void* ptr = nullptr;
        posix_memalign(&ptr, 64, size);
        return static_cast<float*>(ptr);
    }
    
    static void free_aligned(float* ptr) {
        if (ptr) free(ptr);
    }
    
    static size_t compute_size(const std::vector<int64_t>& shape) {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    
    std::vector<TensorStorage*> get_dependents(TensorStorage* storage) {
        std::vector<TensorStorage*> result;
        for (uint64_t dep_id : storage->dependent_ids) {
            auto it = storages_.find(dep_id);
            if (it != storages_.end()) {
                result.push_back(it->second.get());
            }
        }
        return result;
    }
    
    void free_storage(TensorStorage* storage) {
        if (!storage) return;
        
        if (storage->data && !storage->is_view) {
            free_aligned(storage->data);
            active_bytes_ -= storage->capacity;
        }
        
        storage->data = nullptr;
        storage->capacity = 0;
        storage->size = 0;
    }
    
    std::unordered_map<uint64_t, std::unique_ptr<TensorStorage>> storages_;
    uint64_t next_storage_id_;
    size_t active_bytes_;
    size_t peak_bytes_;
};

inline MemoryGraphEngine& get_memory_engine() {
    return MemoryGraphEngine::get_instance();
}

}
