#ifndef RCCL_TEST_UTILS_HPP
#define RCCL_TEST_UTILS_HPP

#include <rccl/rccl.h>

#include <string>

namespace rccl_test {

bool check_nccl_result(
    ncclResult_t result,
    const std::string& operation);

}  // namespace rccl_test

#endif