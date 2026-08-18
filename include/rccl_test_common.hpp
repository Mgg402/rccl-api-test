#ifndef RCCL_TEST_COMMON_HPP
#define RCCL_TEST_COMMON_HPP

#include <iostream>
#include <string>

namespace rccl_test {

void print_test_header(const std::string& test_name);

void print_pass(const std::string& test_name);

void print_fail(
    const std::string& test_name,
    const std::string& message);

}  // namespace rccl_test

#endif