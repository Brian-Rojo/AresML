#pragma once

#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

namespace aresml {

struct ProfileEvent {
    std::string name;
    double duration_ms = 0.0;
    int count = 0;
};

class Profiler {
private:
    static inline Profiler& instance() {
        static Profiler prof;
        return prof;
    }
    
    std::unordered_map<std::string, ProfileEvent> events;
    std::chrono::high_resolution_clock::time_point last_start;
    std::string last_name;
    bool enabled = false;
    
public:
    static void enable() { instance().enabled = true; }
    static void disable() { instance().enabled = false; }
    
    static void start(const std::string& name) {
        if (!instance().enabled) return;
        instance().last_name = name;
        instance().last_start = std::chrono::high_resolution_clock::now();
    }
    
    static void end() {
        if (!instance().enabled) return;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(now - instance().last_start).count();
        
        auto& event = instance().events[instance().last_name];
        event.name = instance().last_name;
        event.duration_ms += duration;
        event.count++;
    }
    
    static void report() {
        if (instance().events.empty()) {
            std::cout << "[PROFILER] No events recorded\n";
            return;
        }
        
        std::cout << "\n=== PROFILER REPORT ===\n";
        std::cout << std::left << std::setw(40) << "Operation" 
                  << std::setw(12) << "Count" 
                  << std::setw(15) << "Total (ms)" 
                  << std::setw(15) << "Avg (ms)" << "\n";
        std::cout << std::string(82, '-') << "\n";
        
        double total_time = 0.0;
        for (auto& event : instance().events) {
            total_time += event.second.duration_ms;
        }
        
        // Sort by duration
        std::vector<std::pair<std::string, ProfileEvent>> sorted(
            instance().events.begin(), instance().events.end());
        std::sort(sorted.begin(), sorted.end(), 
            [](const auto& a, const auto& b) { return a.second.duration_ms > b.second.duration_ms; });
        
        for (auto& [name, event] : sorted) {
            double avg = event.duration_ms / event.count;
            std::cout << std::left << std::setw(40) << name
                      << std::setw(12) << event.count
                      << std::setw(15) << std::fixed << std::setprecision(4) << event.duration_ms
                      << std::setw(15) << std::fixed << std::setprecision(6) << avg << "\n";
        }
        
        std::cout << std::string(82, '-') << "\n";
        std::cout << std::left << std::setw(40) << "TOTAL"
                  << std::setw(12) << ""
                  << std::setw(15) << std::fixed << std::setprecision(4) << total_time << "\n";
        std::cout << "========================\n";
    }
    
    static void reset() {
        instance().events.clear();
    }
};

// RAII scope timer
struct ScopeTimer {
    std::string name;
    
    ScopeTimer(const std::string& n) : name(n) {
        Profiler::start(name);
    }
    
    ~ScopeTimer() {
        Profiler::end();
    }
};

} // namespace aresml

#define PROFILE_SCOPE(name) aresml::ScopeTimer __profile_timer__(name)
#define PROFILE_START(name) aresml::Profiler::start(name)
#define PROFILE_END() aresml::Profiler::end()
#define PROFILE_ENABLE() aresml::Profiler::enable()
#define PROFILE_DISABLE() aresml::Profiler::disable()
#define PROFILE_REPORT() aresml::Profiler::report()
#define PROFILE_RESET() aresml::Profiler::reset()
