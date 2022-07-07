/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>
#include <faiss/pipe/PipeKernelImpl.cuh>

namespace faiss {
namespace gpu {

IVF_INTERLEAVED_IMPL(128, 1024, 8)

//KERNEL_COMPUTE_C_IMPL(128, 1024, 8)
KERNEL_COMPUTE_IMPL(128, 1024, 8)

}
} // namespace faiss
