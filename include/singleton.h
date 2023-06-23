#pragma once

#define SINGLETON(class_name, params) __SINGLETON_IMPL(class_name, params)

#define __SINGLETON_IMPL(class_name, params) \
    public: \
        void operator=(const class_name &) = delete; \
        class_name(const class_name &) = delete; \
        static class_name &Instance() { \
            static class_name instance; \
            return instance; \
        } \
    private: \
        class_name(params);

