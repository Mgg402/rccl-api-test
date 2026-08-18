#include <gtest/gtest.h>
#include <rccl/rccl.h>

TEST(RcclCommLifecycleTest, CommDestroy)
{
    ncclComm_t comm = nullptr;
    int dev = 0;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);
    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST(RcclCommLifecycleTest, CommFinalize)
{
    ncclComm_t comm = nullptr;
    int dev = 0;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);
    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(ncclCommFinalize(comm), ncclSuccess);
}

TEST(RcclCommLifecycleTest, CommAbort)
{
    ncclComm_t comm = nullptr;
    int dev = 0;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);
    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(ncclCommAbort(comm), ncclSuccess);
}