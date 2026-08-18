#include "rccl_comm_utils.hpp"

namespace rccl_test {

NcclCommGuard::NcclCommGuard()
    : comm_(nullptr) {
}

NcclCommGuard::~NcclCommGuard() {
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
}

ncclComm_t NcclCommGuard::get() const {
    return comm_;
}

void NcclCommGuard::reset(ncclComm_t comm) {
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }

    comm_ = comm;
}

}  // namespace rccl_test