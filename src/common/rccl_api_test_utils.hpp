#pragma once

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

namespace rccl_test {

class CommGuard {
public:
    CommGuard() = default;

    ~CommGuard()
    {
        if (comm_ != nullptr) {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }
    }

    CommGuard(const CommGuard&) = delete;
    CommGuard& operator=(const CommGuard&) = delete;

    ncclComm_t get() const
    {
        return comm_;
    }

    ncclComm_t* ptr()
    {
        return &comm_;
    }

private:
    ncclComm_t comm_ = nullptr;
};

inline ncclComm_t createSingleRankComm()
{
    ncclUniqueId id{};

    EXPECT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    ncclComm_t comm = nullptr;

    EXPECT_EQ(
        ncclCommInitRank(&comm, 1, id, 0),
        ncclSuccess
    );

    EXPECT_NE(comm, nullptr);

    return comm;
}

inline void destroyComm(ncclComm_t comm)
{
    if (comm != nullptr) {
        EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
    }
}

template <typename T>
T* allocDevice(size_t count)
{
    T* ptr = nullptr;

    EXPECT_EQ(
        hipMalloc(reinterpret_cast<void**>(&ptr),
                  sizeof(T) * count),
        hipSuccess
    );

    return ptr;
}

template <typename T>
void freeDevice(T* ptr)
{
    if (ptr != nullptr) {
        EXPECT_EQ(hipFree(ptr), hipSuccess);
    }
}

template <typename T>
void copyToDevice(T* dst, const T* src, size_t count)
{
    EXPECT_EQ(
        hipMemcpy(
            dst,
            src,
            sizeof(T) * count,
            hipMemcpyHostToDevice
        ),
        hipSuccess
    );
}

template <typename T>
void copyToHost(T* dst, const T* src, size_t count)
{
    EXPECT_EQ(
        hipMemcpy(
            dst,
            src,
            sizeof(T) * count,
            hipMemcpyDeviceToHost
        ),
        hipSuccess
    );
}

inline void synchronize()
{
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

} // namespace rccl_test