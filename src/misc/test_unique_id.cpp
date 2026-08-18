#include <gtest/gtest.h>

#include <cstring>

#include <rccl/rccl.h>

TEST(RcclUniqueIdTest, GetUniqueId)
{
    ncclUniqueId id{};

    ncclResult_t result = ncclGetUniqueId(&id);

    EXPECT_EQ(result, ncclSuccess);
}

TEST(RcclUniqueIdTest, GetUniqueIdNull)
{
    ncclResult_t result = ncclGetUniqueId(nullptr);

    EXPECT_EQ(result, ncclInvalidArgument);
}

TEST(RcclUniqueIdTest, GetUniqueIdWritesOutput)
{
    ncclUniqueId id{};

    ncclResult_t result = ncclGetUniqueId(&id);

    ASSERT_EQ(result, ncclSuccess);

    bool nonZero = false;

    for (size_t i = 0; i < sizeof(id.internal); ++i) {
        if (id.internal[i] != 0) {
            nonZero = true;
            break;
        }
    }

    EXPECT_TRUE(nonZero);
}

TEST(RcclUniqueIdTest, GetUniqueIdMultipleTimes)
{
    ncclUniqueId id1{};
    ncclUniqueId id2{};

    ASSERT_EQ(ncclGetUniqueId(&id1), ncclSuccess);
    ASSERT_EQ(ncclGetUniqueId(&id2), ncclSuccess);

    EXPECT_NE(
        std::memcmp(&id1, &id2, sizeof(ncclUniqueId)),
        0
    );
}