 RCCL API Test
│
├── Communicator
│   ├── RcclCommApiTest
│   │   ├── InitRank / Destroy
│   │   ├── Finalize
│   │   ├── GetAsyncError
│   │   ├── Count
│   │   ├── CuDevice
│   │   ├── UserRank
│   │   ├── Abort
│   │   ├── InitAll
│   │   ├── InitRankConfig
│   │   ├── Register / Deregister
│   │   └── Split
│   │
│   ├── RcclCommShrinkTest
│   │   └── Shrink
│   │
│   └── RcclCommWindowApiTest
│       └── Window Register / Deregister
│
├── Collective
│   └── RcclCollectiveApiTest
│       ├── AllReduce
│       ├── AllGather
│       ├── Broadcast
│       ├── Reduce
│       ├── ReduceScatter
│       ├── Scatter
│       ├── Gather
│       ├── AllToAll
│       ├── AllToAllv
│       ├── Send / Recv
│       └── Group Start / End
│
├── Memory
│   └── RcclMemoryApiTest
│       ├── MemAlloc
│       └── MemFree
│
├── Datatype / RedOp
│   └── RcclRedOpApiTest
│       └── CreatePreMulSum / Destroy
│
├── Error
│   └── RcclErrorTest
│       ├── GetErrorString
│       └── GetLastError
│
├── Misc
│   ├── RcclVersionTest
│   │   └── GetVersion
│   └── RcclUniqueIdTest
│       └── GetUniqueId
│
├── MSCCL
│   └── RcclMscclApiTest
│       └── Load / Run / Unload Algo
│
└── Extended Collective
    └── RcclAllReduceBiasApiTest
        └── AllReduceWithBias
 

 
 
                    RCCL API Test
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ↓                  ↓                  ↓
 Communicator          Collective          Utility
       │                  │                  │
       │                  │                  ├─ GetVersion
       │                  │                  ├─ Error
       │                  │                  └─ RedOp
       │                  │
       ├─ Init             ├─ AllReduce
       ├─ Destroy          ├─ Broadcast
       ├─ Finalize         ├─ Reduce
       ├─ Abort            ├─ Gather
       ├─ Split            ├─ Scatter
       ├─ Shrink           ├─ AllGather
       ├─ Count            ├─ ReduceScatter
       ├─ UserRank         ├─ AllToAll
       └─ CuDevice         └─ AllToAllv

       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
     Memory              P2P                MSCCL
       │                  │                  │
       ├─ MemAlloc        ├─ Send            ├─ Load
       ├─ MemFree         └─ Recv            ├─ Run
       ├─ Register                           └─ Unload
       └─ Window



Comm
├── ncclCommGetAsyncError
├── ncclCommCount
├── ncclCommInitRank
├── ncclCommDestroy
├── ncclCommFinalize
├── ncclGetUniqueId
├── ncclCommInitAll
├── ncclCommInitRankConfig
├── ncclCommAbort
├── ncclCommSplit
├── ncclCommCuDevice
├── ncclCommUserRank
└── ncclCommShrink

Collective
├── ncclAllGather
├── ncclAllReduce
├── ncclAllToAll
├── ncclAllToAllv
├── ncclBroadcast
├── ncclGather
├── ncclReduce
└── ncclReduceScatter

P2P
├── ncclSend
└── ncclRecv

Group
├── ncclGroupStart
└── ncclGroupEnd

Memory
├── ncclMemAlloc
└── ncclMemFree

Register
├── ncclCommRegister
└── ncclCommDeregister

RedOp
├── ncclRedOpCreatePreMulSum
└── ncclRedOpDestroy

Error
├── ncclGetErrorString
├── ncclGetLastError
└── ncclCommGetAsyncError

Misc
├── ncclGetVersion
└── ncclAllReduceWithBias

MSCCL
├── mscclLoadAlgo
├── mscclRunAlgo
└── mscclUnloadAlgo

Window
├── ncclCommWindowRegister
└── ncclCommWindowDeregister