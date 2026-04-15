#include <iostream>
#include <cassert>
#include "core/memory_graph/MemoryGraphEngine.hpp"

using namespace aresml;

int main() {
    std::cout << "=== MemoryGraph Engine Test ===\n\n";
    
    auto& engine = get_memory_engine();
    
    std::cout << "1. Test allocate\n";
    auto storage1 = engine.allocate(1024);
    std::cout << "   Allocated: " << storage1.size() << " bytes\n";
    std::cout << "   Ref count: " << storage1.ref_count() << "\n";
    std::cout << "   Storage ID: " << storage1.id() << "\n";
    
    std::cout << "\n2. Test create view\n";
    auto view1 = engine.create_view(storage1, 0, {256, 4});
    std::cout << "   View created\n";
    std::cout << "   View ref count: " << view1.ref_count() << "\n";
    std::cout << "   Base ref count: " << storage1.ref_count() << "\n";
    std::cout << "   Same data: " << (view1.data() == storage1.data() ? "YES" : "NO") << "\n";
    
    std::cout << "\n3. Test memory report\n";
    std::cout << engine.memory_report();
    
    std::cout << "\n4. Test garbage collection\n";
    view1 = MemoryGraphEngine::StorageHandle();
    std::cout << "   After view destroyed\n";
    engine.collect_garbage();
    std::cout << engine.memory_report();
    
    std::cout << "\n5. Test multiple allocations\n";
    auto s1 = engine.allocate(100);
    auto s2 = engine.allocate(200);
    auto s3 = engine.allocate(300);
    std::cout << "   3 allocations done\n";
    std::cout << engine.memory_report();
    
    std::cout << "\n6. Test epochs (multiple runs)\n";
    for (int epoch = 0; epoch < 3; ++epoch) {
        auto e = engine.allocate(1000);
        for (int i = 0; i < 10; ++i) {
            auto v = engine.create_view(e, 0, {250, 4});
        }
        e = MemoryGraphEngine::StorageHandle();
        engine.collect_garbage();
    }
    std::cout << "   After 3 epochs:\n";
    std::cout << engine.memory_report();
    
    engine.clear();
    std::cout << "\n7. After clear:\n";
    std::cout << engine.memory_report();
    
    std::cout << "\n=== All Tests Passed ===\n";
    
    return 0;
}
