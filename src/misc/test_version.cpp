#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <iostream>

TEST(RcclVersionTest, GetVersion)
{
    int version = 0;

    ncclResult_t result = ncclGetVersion(&version);

    ASSERT_EQ(result, ncclSuccess);
    EXPECT_GT(version, 0);

    std::cout << "RCCL version: " << version << std::endl;
}
