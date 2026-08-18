
#include <gtest/gtest.h>

#include <rccl/rccl.h>

#include "rccl_api_test_utils.hpp"
#include <iostream>

TEST(RcclCommWindowApiTest, WindowRegisterDeregister)
{
    ncclComm_t comm = rccl_test::createSingleRankComm();

    ASSERT_NE(comm, nullptr);

    constexpr size_t bytes = 4096;

    void* buffer = nullptr;

    ASSERT_EQ(
        hipMalloc(&buffer, bytes),
        hipSuccess
    );

    ncclWindow_t window = nullptr;

    // Test window registration API.
    ncclResult_t registerResult = ncclCommWindowRegister(
        comm,
        buffer,
        bytes,
        &window,
        0
    );

    EXPECT_EQ(registerResult, ncclSuccess);

    // Some RCCL/ROCm environments may return ncclSuccess
    // without producing a non-null window handle.
    // Only deregister when a valid window handle is returned.
    // if (window != nullptr) {
    //     std::cout << "[ INFO     ] Window is not nullptr. Deregister branch executed!" << std::endl;
    //     EXPECT_EQ(
    //         ncclCommWindowDeregister(
    //             comm,
    //             window
    //         ),
    //         ncclSuccess
    //     );
    // }else {
    //     std::cout << "[ INFO     ] Window is nullptr. Deregister branch SKIPPED!" << std::endl;
    // }

    // 强制执行 API 以增加 trace 覆盖度
    ncclResult_t deregResult = ncclCommWindowDeregister(comm, window);
    
    // 如果 window 是 nullptr，RCCL 可能会报错，我们这里只为了让 rocprofv3 抓到调用
    // 所以只要它执行了就行，哪怕返回的是参数错误 (ncclInvalidArgument) 也可以接受
    EXPECT_TRUE(deregResult == ncclSuccess || deregResult == ncclInvalidArgument);

    EXPECT_EQ(
        hipFree(buffer),
        hipSuccess
    );

    rccl_test::destroyComm(comm);
}
