#include <gtest/gtest.h>

#include <rccl/rccl.h>
#include <cstring>

#include "rccl_api_test_utils.hpp"


TEST(RcclMemoryApiTest, MemAlloc)
{
    void* ptr = nullptr;

    EXPECT_EQ(
        ncclMemAlloc(
            &ptr,
            4096
        ),
        ncclSuccess
    );

    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(
        ncclMemFree(ptr),
        ncclSuccess
    );
}


TEST(RcclMemoryApiTest, MemAllocAndFreeMultiple)
{
    constexpr size_t size = 4096;

    void* ptr1 = nullptr;
    void* ptr2 = nullptr;

    ASSERT_EQ(
        ncclMemAlloc(&ptr1, size),
        ncclSuccess
    );

    ASSERT_EQ(
        ncclMemAlloc(&ptr2, size),
        ncclSuccess
    );

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    EXPECT_NE(ptr1, ptr2);

    EXPECT_EQ(
        ncclMemFree(ptr1),
        ncclSuccess
    );

    EXPECT_EQ(
        ncclMemFree(ptr2),
        ncclSuccess
    );
}