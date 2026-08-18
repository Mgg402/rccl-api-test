#include <gtest/gtest.h>
#include <rccl/rccl.h>

TEST(RcclCommInfoTest, CommCount)
{
    ncclComm_t comm = nullptr;
    int dev = 0;
    int count = 0;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);
    ASSERT_NE(comm, nullptr);

    ncclResult_t result = ncclCommCount(comm, &count);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(count, 1);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST(RcclCommInfoTest, CommCuDevice)
{
    ncclComm_t comm = nullptr;
    int dev = 0;
    int device = -1;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);

    ncclResult_t result = ncclCommCuDevice(comm, &device);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(device, 0);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST(RcclCommInfoTest, CommUserRank)
{
    ncclComm_t comm = nullptr;
    int dev = 0;
    int rank = -1;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);

    ncclResult_t result = ncclCommUserRank(comm, &rank);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(rank, 0);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST(RcclCommInfoTest, CommGetAsyncError)
{
    ncclComm_t comm = nullptr;
    int dev = 0;
    ncclResult_t asyncError = ncclSuccess;

    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);

    ncclResult_t result = ncclCommGetAsyncError(comm, &asyncError);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(asyncError, ncclSuccess);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}