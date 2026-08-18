#include <gtest/gtest.h>

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include "rccl_api_test_utils.hpp"

TEST(RcclAllReduceBiasApiTest, AllReduceWithBias)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();
    ASSERT_NE(comm, nullptr);

    constexpr size_t count = 1024;
    constexpr size_t bytes = count * sizeof(float);

    float* sendbuff = nullptr;
    float* recvbuff = nullptr;
    float* acc = nullptr;

    ASSERT_EQ(hipMalloc(&sendbuff, bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&recvbuff, bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&acc, bytes), hipSuccess);

    EXPECT_EQ(
        ncclAllReduceWithBias(
            sendbuff,
            recvbuff,
            count,
            ncclFloat32,
            ncclSum,
            comm,
            nullptr,
            acc
        ),
        ncclSuccess
    );

    EXPECT_EQ(hipFree(sendbuff), hipSuccess);
    EXPECT_EQ(hipFree(recvbuff), hipSuccess);
    EXPECT_EQ(hipFree(acc), hipSuccess);

    rccl_test::destroyComm(comm);
}