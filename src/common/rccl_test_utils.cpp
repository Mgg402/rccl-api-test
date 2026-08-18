#include "rccl_test_utils.hpp"

#include <iostream>

namespace rccl_test {

bool check_nccl_result(
    ncclResult_t result,
    const std::string& operation) {

    if (result == ncclSuccess) {
        return true;
    }

    std::cerr
        << "[RCCL ERROR] "
        << operation
        << " failed: "
        << ncclGetErrorString(result)
        << std::endl;

    return false;
}

}  // namespace rccl_test