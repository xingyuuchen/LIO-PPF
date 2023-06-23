#include <iostream>
#include <iomanip>
#include <glog/logging.h>
#include "timer.h"


Timer::Timer() = default;

void Timer::StartRecord() {
    start_tick_count_.emplace(GetTimePoint());
}

void Timer::EndRecord(const std::string &func_name) {
    double duration = EndSingleRecord();
    
    if (records_.find(func_name) != records_.end()) {
        records_[func_name].time_usage_in_ms_.emplace_back(duration);
    } else {
        records_.insert({func_name, TimerRecord(func_name, duration)});
    }
}

double Timer::EndSingleRecord() {
    auto now = GetTimePoint();
    LOG_ASSERT(!start_tick_count_.empty());
    auto start = start_tick_count_.top();
    start_tick_count_.pop();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
            now - start).count() * 1000;
    return duration;
}

void Timer::PrintAll() {
    LOG_ASSERT(start_tick_count_.empty());
    if (records_.empty()) {
        return;
    }
    for (const auto& r : records_) {
        std::cout << ">  [ " << r.first << " ] mean: "
                  << std::accumulate(r.second.time_usage_in_ms_.begin(), r.second.time_usage_in_ms_.end(), 0.0) /
                     double(r.second.time_usage_in_ms_.size())
                  << " ms, n_times: " << r.second.time_usage_in_ms_.size() << std::endl;
    }
}

void Timer::DumpIntoFile(const std::string& file_name) {
    std::ofstream ofs(file_name, std::ios::out);
    if (!ofs.is_open()) {
        LOG(ERROR) << "Failed to open file: " << file_name;
        return;
    }
    size_t min_length = 1e8;
    for (const auto& iter : records_) {
        if (iter.second.time_usage_in_ms_.size() < min_length) {
            min_length = iter.second.time_usage_in_ms_.size();
        }
    }

    for (int i = 0; i < min_length; ++i) {
        double time_scan = 0;
        for (const auto& record : records_) {
            time_scan += record.second.time_usage_in_ms_[i];
        }
        ofs << std::fixed << std::setprecision(2) << time_scan << '\n';
    }
    ofs.close();
}


