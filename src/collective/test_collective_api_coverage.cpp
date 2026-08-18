#include <gtest/gtest.h>

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include "rccl_api_test_utils.hpp"


namespace {

struct CommFixture {
    ncclComm_t comm = nullptr;

    CommFixture()
    {
        comm = rccl_test::createSingleRankComm();
    }

    ~CommFixture()
    {
        rccl_test::destroyComm(comm);
    }
};

} // namespace


TEST(RcclCollectiveApiTest, AllReduce)
{
    CommFixture fixture;

    float hostInput = 1.0f;
    float hostOutput = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &hostInput, 1);

    EXPECT_EQ(
        ncclAllReduce(
            input,
            output,
            1,
            ncclFloat32,
            ncclSum,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&hostOutput, output, 1);

    EXPECT_FLOAT_EQ(hostOutput, 1.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, AllGather)
{
    CommFixture fixture;

    float inputHost = 1.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclAllGather(
            input,
            output,
            1,
            ncclFloat32,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 1.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, Broadcast)
{
    CommFixture fixture;

    float inputHost = 7.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclBroadcast(
            input,
            output,
            1,
            ncclFloat32,
            0,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 7.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, Reduce)
{
    CommFixture fixture;

    float inputHost = 3.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclReduce(
            input,
            output,
            1,
            ncclFloat32,
            ncclSum,
            0,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 3.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, ReduceScatter)
{
    CommFixture fixture;

    float inputHost = 5.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclReduceScatter(
            input,
            output,
            1,
            ncclFloat32,
            ncclSum,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 5.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, Scatter)
{
    CommFixture fixture;

    float inputHost = 6.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclScatter(
            input,
            output,
            1,
            ncclFloat32,
            0,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 6.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, Gather)
{
    CommFixture fixture;

    float inputHost = 8.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclGather(
            input,
            output,
            1,
            ncclFloat32,
            0,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 8.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, AllToAll)
{
    CommFixture fixture;

    float inputHost = 9.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclAllToAll(
            input,
            output,
            1,
            ncclFloat32,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 9.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, AllToAllv)
{
    CommFixture fixture;

    float inputHost = 10.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    size_t sendCounts[1] = {1};
    size_t sendDispls[1] = {0};
    size_t recvCounts[1] = {1};
    size_t recvDispls[1] = {0};

    EXPECT_EQ(
        ncclAllToAllv(
            input,
            sendCounts,
            sendDispls,
            output,
            recvCounts,
            recvDispls,
            ncclFloat32,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 10.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, SendRecv)
{
    CommFixture fixture;

    float inputHost = 11.0f;
    float outputHost = 0.0f;

    float* input = rccl_test::allocDevice<float>(1);
    float* output = rccl_test::allocDevice<float>(1);

    rccl_test::copyToDevice(input, &inputHost, 1);

    EXPECT_EQ(
        ncclGroupStart(),
        ncclSuccess
    );

    EXPECT_EQ(
        ncclSend(
            input,
            1,
            ncclFloat32,
            0,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    EXPECT_EQ(
        ncclRecv(
            output,
            1,
            ncclFloat32,
            0,
            fixture.comm,
            nullptr
        ),
        ncclSuccess
    );

    EXPECT_EQ(
        ncclGroupEnd(),
        ncclSuccess
    );

    rccl_test::synchronize();

    rccl_test::copyToHost(&outputHost, output, 1);

    EXPECT_FLOAT_EQ(outputHost, 11.0f);

    rccl_test::freeDevice(input);
    rccl_test::freeDevice(output);
}


TEST(RcclCollectiveApiTest, GroupStartEnd)
{
    EXPECT_EQ(
        ncclGroupStart(),
        ncclSuccess
    );

    EXPECT_EQ(
        ncclGroupEnd(),
        ncclSuccess
    );
}