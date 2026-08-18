#ifndef RCCL_COMM_UTILS_HPP
#define RCCL_COMM_UTILS_HPP

#include <rccl/rccl.h>

namespace rccl_test {

class NcclCommGuard {
public:
    NcclCommGuard();
    ~NcclCommGuard();

    NcclCommGuard(const NcclCommGuard&) = delete;
    NcclCommGuard& operator=(const NcclCommGuard&) = delete;

    ncclComm_t get() const;

    void reset(ncclComm_t comm);

private:
    ncclComm_t comm_;
};

}  // namespace rccl_test

#endif