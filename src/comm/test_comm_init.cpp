#include <gtest/gtest.h>
#include <rccl/rccl.h>

TEST(RcclCommInitTest, CommInitAll)
{
    ncclComm_t comm = nullptr;
    int dev = 0;

    ncclResult_t result =
        ncclCommInitAll(&comm, 1, &dev);

    ASSERT_EQ(result, ncclSuccess);
    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST(RcclCommInitTest, CommInitRank)
{
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    ncclComm_t comm = nullptr;

    ncclResult_t result =
        ncclCommInitRank(
            &comm,
            1,
            id,
            0
        );

    ASSERT_EQ(result, ncclSuccess);
    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST(RcclCommInitTest, CommInitRankConfig)
{
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    ncclComm_t comm = nullptr;

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    ncclResult_t result =
        ncclCommInitRankConfig(
            &comm,
            1,
            id,
            0,
            &config
        );

    ASSERT_EQ(result, ncclSuccess);
    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}