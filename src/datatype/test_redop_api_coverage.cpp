#include <gtest/gtest.h>

#include <rccl/rccl.h>

#include "rccl_api_test_utils.hpp"


TEST(RcclRedOpApiTest, CreatePreMulSumAndDestroy)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    ncclRedOp_t op = ncclSum;

    float scalar = 2.0f;

    EXPECT_EQ(
        ncclRedOpCreatePreMulSum(
            &op,
            &scalar,
            ncclFloat32,
            ncclScalarHostImmediate,
            comm
        ),
        ncclSuccess
    );

    EXPECT_EQ(
        ncclRedOpDestroy(
            op,
            comm
        ),
        ncclSuccess
    );

    rccl_test::destroyComm(comm);
}