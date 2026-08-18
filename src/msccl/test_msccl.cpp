#include <gtest/gtest.h>

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include "rccl_api_test_utils.hpp"

TEST(RcclMscclApiTest, MscclLifecycle)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();
    ASSERT_NE(comm, nullptr);

    mscclAlgoHandle_t handle = 0;

    EXPECT_EQ(
        mscclLoadAlgo(
            "deprecated_test_algo",
            &handle,
            0
        ),
        ncclSuccess
    );

    constexpr size_t count = 1;

    size_t sendCounts[1] = {count};
    size_t sendDispls[1] = {0};
    size_t recvCounts[1] = {count};
    size_t recvDispls[1] = {0};

    float* sendbuff = nullptr;
    float* recvbuff = nullptr;

    ASSERT_EQ(
        hipMalloc(&sendbuff, sizeof(float)),
        hipSuccess
    );

    ASSERT_EQ(
        hipMalloc(&recvbuff, sizeof(float)),
        hipSuccess
    );

    EXPECT_EQ(
        mscclRunAlgo(
            sendbuff,
            sendCounts,
            sendDispls,
            recvbuff,
            recvCounts,
            recvDispls,
            count,
            ncclFloat32,
            0,
            0,
            ncclSum,
            handle,
            comm,
            nullptr
        ),
        ncclSuccess
    );

    EXPECT_EQ(
        mscclUnloadAlgo(handle),
        ncclSuccess
    );

    EXPECT_EQ(hipFree(sendbuff), hipSuccess);
    EXPECT_EQ(hipFree(recvbuff), hipSuccess);

    rccl_test::destroyComm(comm);
}