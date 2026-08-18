#include <gtest/gtest.h>
#include <iostream>
#include <rccl/rccl.h>

#include "rccl_api_test_utils.hpp"

//验证 ncclCommShrink 在 excludeRanksList == nullptr、excludeRanksCount == 0 的非法参数情况下，能够正确返回 ncclInvalidArgument。
TEST(RcclCommShrinkTest, ShrinkInvalidArguments)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    ncclComm_t newComm = nullptr;

    ncclResult_t result = ncclCommShrink(
        comm,
        nullptr,
        0,
        &newComm,
        nullptr,
        0
    );

    EXPECT_EQ(result, ncclInvalidArgument);
    EXPECT_EQ(newComm, nullptr);

    rccl_test::destroyComm(comm);
}