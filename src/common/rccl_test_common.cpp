#include "rccl_test_common.hpp"

namespace rccl_test {

void print_test_header(const std::string& test_name) {
    std::cout
        << "========================================\n"
        << "TEST: " << test_name << "\n"
        << "========================================\n";
}

void print_pass(const std::string& test_name) {
    std::cout
        << "[PASS] "
        << test_name
        << std::endl;
}

void print_fail(
    const std::string& test_name,
    const std::string& message) {

    std::cerr
        << "[FAIL] "
        << test_name
        << ": "
        << message
        << std::endl;
}

}  // namespace rccl_test