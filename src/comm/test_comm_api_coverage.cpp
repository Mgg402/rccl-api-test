#include <gtest/gtest.h>

#include <rccl/rccl.h>

#include "rccl_api_test_utils.hpp"


TEST(RcclCommApiTest, CommInitRankAndDestroy)
{
    ncclUniqueId id{};

    ASSERT_EQ(
        ncclGetUniqueId(&id),
        ncclSuccess
    );

    ncclComm_t comm = nullptr;

    ASSERT_EQ(
        ncclCommInitRank(
            &comm,
            1,
            id,
            0
        ),
        ncclSuccess
    );

    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(
        ncclCommDestroy(comm),
        ncclSuccess
    );
}


TEST(RcclCommApiTest, CommFinalize)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(
        ncclCommFinalize(comm),
        ncclSuccess
    );

    /*
     * ncclCommFinalize() is followed by ncclCommDestroy()
     * to release the communicator object.
     */
    EXPECT_EQ(
        ncclCommDestroy(comm),
        ncclSuccess
    );
}


TEST(RcclCommApiTest, CommGetAsyncError)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    ncclResult_t asyncError = ncclSuccess;

    EXPECT_EQ(
        ncclCommGetAsyncError(
            comm,
            &asyncError
        ),
        ncclSuccess
    );

    EXPECT_EQ(
        asyncError,
        ncclSuccess
    );

    rccl_test::destroyComm(comm);
}


TEST(RcclCommApiTest, CommCount)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    int count = 0;

    EXPECT_EQ(
        ncclCommCount(
            comm,
            &count
        ),
        ncclSuccess
    );

    EXPECT_EQ(count, 1);

    rccl_test::destroyComm(comm);
}


TEST(RcclCommApiTest, CommCuDevice)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    int device = -1;

    EXPECT_EQ(
        ncclCommCuDevice(
            comm,
            &device
        ),
        ncclSuccess
    );

    EXPECT_GE(device, 0);

    rccl_test::destroyComm(comm);
}


TEST(RcclCommApiTest, CommUserRank)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    int rank = -1;

    EXPECT_EQ(
        ncclCommUserRank(
            comm,
            &rank
        ),
        ncclSuccess
    );

    EXPECT_EQ(rank, 0);

    rccl_test::destroyComm(comm);
}


TEST(RcclCommApiTest, CommAbort)
{
    ncclUniqueId id{};

    ASSERT_EQ(
        ncclGetUniqueId(&id),
        ncclSuccess
    );

    ncclComm_t comm = nullptr;

    ASSERT_EQ(
        ncclCommInitRank(
            &comm,
            1,
            id,
            0
        ),
        ncclSuccess
    );

    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(
        ncclCommAbort(comm),
        ncclSuccess
    );
}


TEST(RcclCommApiTest, CommInitAll)
{
    int device = 0;
    ncclComm_t comm = nullptr;

    EXPECT_EQ(
        ncclCommInitAll(
            &comm,
            1,
            &device
        ),
        ncclSuccess
    );

    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(
        ncclCommDestroy(comm),
        ncclSuccess
    );
}


TEST(RcclCommApiTest, CommInitRankConfig)
{
    ncclUniqueId id{};

    ASSERT_EQ(
        ncclGetUniqueId(&id),
        ncclSuccess
    );

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    ncclComm_t comm = nullptr;

    ASSERT_EQ(
        ncclCommInitRankConfig(
            &comm,
            1,
            id,
            0,
            &config
        ),
        ncclSuccess
    );

    ASSERT_NE(comm, nullptr);

    EXPECT_EQ(
        ncclCommDestroy(comm),
        ncclSuccess
    );
}


TEST(RcclCommApiTest, CommRegisterDeregister)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    constexpr size_t bytes = 1024;

    void* buffer = nullptr;

    ASSERT_EQ(
        hipMalloc(&buffer, bytes),
        hipSuccess
    );

    void* handle = nullptr;

    EXPECT_EQ(
        ncclCommRegister(
            comm,
            buffer,
            bytes,
            &handle
        ),
        ncclSuccess
    );

    ASSERT_NE(handle, nullptr);

    EXPECT_EQ(
        ncclCommDeregister(
            comm,
            handle
        ),
        ncclSuccess
    );

    EXPECT_EQ(
        hipFree(buffer),
        hipSuccess
    );

    rccl_test::destroyComm(comm);
}


TEST(RcclCommApiTest, CommSplit)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    ncclComm_t newComm = nullptr;

    EXPECT_EQ(
        ncclCommSplit(
            comm,
            0,
            0,
            &newComm,
            nullptr
        ),
        ncclSuccess
    );

    ASSERT_NE(newComm, nullptr);

    EXPECT_EQ(
        ncclCommDestroy(newComm),
        ncclSuccess
    );

    rccl_test::destroyComm(comm);
}