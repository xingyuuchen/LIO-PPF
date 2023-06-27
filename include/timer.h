#pragma once
#include <chrono>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <stack>
#include <iostream>
#include "singleton.h"

#ifdef DEBUG
#define D_RECORD_TIME_START Timer::Instance().StartRecord()
#define D_RECORD_TIME_END(FUNC_NAME) Timer::Instance().EndRecord(FUNC_NAME)

#else
#define D_RECORD_TIME_START do {} while (0)
#define D_RECORD_TIME_END(FUNC_NAME) do {} while (0)
#endif


class Timer {
    
    SINGLETON(Timer, )
  
  public:
    struct TimerRecord {
        TimerRecord() = default;
        TimerRecord(const std::string& name, double time_usage) {
            func_name_ = name;
            time_usage_in_ms_.emplace_back(time_usage);
        }
        std::string func_name_;
        std::vector<double> time_usage_in_ms_;
    };
    
    
    void StartRecord();
    
    void EndRecord(const std::string& func_name);
    
    double EndSingleRecord();
    
    void PrintAll();

    void DumpIntoFile(const std::string& file_name);
    
  private:
    using TimePoint = std::chrono::system_clock::time_point;
    
    static inline TimePoint GetTimePoint() {
//    using namespace std::chrono;
//    time_point<std::chrono::system_clock, milliseconds> tp =
//            time_point_cast<milliseconds>(system_clock::now());
//    return tp.time_since_epoch().count();
#ifdef __APPLE__
        return std::chrono::system_clock::now();
#else
        return std::chrono::high_resolution_clock::now();
#endif
    }

  private:
    std::stack<TimePoint> start_tick_count_;
    std::map<std::string, TimerRecord> records_;
    
};


