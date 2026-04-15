#include "Backend.hpp"
#include <memory>

namespace aresml {
namespace backend {

// Global backend instance
static std::unique_ptr<Backend> g_backend;

Backend& get_backend() {
    if (!g_backend) {
        // Default to CPU backend (will be initialized on first use)
        throw std::runtime_error("No backend set. Call set_backend() first.");
    }
    return *g_backend;
}

void set_backend(std::unique_ptr<Backend> backend) {
    if (!backend) {
        throw std::runtime_error("Cannot set null backend");
    }
    g_backend = std::move(backend);
}

} // namespace backend
} // namespace aresml
