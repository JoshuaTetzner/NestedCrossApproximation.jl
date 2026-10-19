# DISABLED: not declared under [extensions] in Project.toml until the BEAST GPU
# extension is released. To re-enable, add CUDA to [weakdeps]/[compat]/[extras]
# and `NestedCrossApproximationCUDAExt = "CUDA"` to [extensions].
module NestedCrossApproximationCUDAExt

using NestedCrossApproximation
using AdaptiveCrossApproximation
using CUDA
using CUDA.CUBLAS: getrf_strided_batched!, getrs_strided_batched!

# GPU backend for BatchedBottomUp: assemble_level returns a device-resident
# stack per device, which selects concurrent GPU ACA and batched LU compression.
# Each device owns its shard, working copy and solve tensors. Only the finished
# bases, transfers and pivot metadata are downloaded. Large host storage loops
# use the caller's scheduler.
include("distributedblockstack.jl")
include("compresslevel.jl")

end
