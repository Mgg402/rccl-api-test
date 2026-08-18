#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "rccl_api_test_utils.hpp"
#include <iostream>


TEST(RcclErrorTest, GetErrorString)
{
    const char* error = ncclGetErrorString(ncclSuccess);

    ASSERT_NE(error, nullptr);
    EXPECT_STREQ(error, "no error");
}

TEST(RcclErrorTest, GetLastError)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    const char* error = ncclGetLastError(comm);

    ASSERT_NE(error, nullptr);

    rccl_test::destroyComm(comm);
}