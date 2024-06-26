/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <thrust/device_vector.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/Comparators.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/PipeTensor.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/WarpPackedBits.cuh>
#include <faiss/gpu/PipeGpuResources.h>

namespace faiss {
namespace gpu {


/// First pass kernel to compute k best vectors for
/// (query, cluster)
template <
        typename Codec,
        typename Metric,
        int ThreadsPerBlock, //128
        int NumWarpQ, //32
        int NumThreadQ, //2
        bool Residual>
__global__ void KernelCompute(
        int d,
        int k,                         
        PipeTensor<int, 1, true> queryids,  
        PipeTensor<float, 2, true> queries,        
        PipeTensor<int, 2, true> query_cluster_matrix,          
        PipeTensor<int, 3, true> best_indices,                  
        PipeTensor<float, 3, true> best_distances,                 
        void** allListData,                     
        int* listLengths,          
        Metric metric,
        Codec codec) {
    extern __shared__ float smem[];

    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    int queryno = blockIdx.y;
    auto uu = queryids(queryno);
    int queryId = queryids(queryno);

    int probeId = blockIdx.x;
    int splitId = blockIdx.z;
    int listId = query_cluster_matrix(queryno)[probeId];
    int numVecs = listLengths[listId];
    int split = gridDim.z;
    int splitStart = numVecs * splitId / split;
    splitStart = utils::roundDown(splitStart, 32);
    int splitEnd = numVecs * (splitId + 1) / split;
    splitEnd = utils::roundDown(splitEnd, 32);
    if (splitId == split - 1) {
        splitEnd = numVecs;
    }

    int splitCnt = splitEnd - splitStart;
    

    // Safety guard in case NaNs in input cause no list ID to be generated, or
    // we have more nprobe than nlist
    if (listId == -1) {
        return;
    }

    int dim = queries.getSize(1);

    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
    int laneId = threadIdx.x % kWarpSize;
    int warpId = threadIdx.x / kWarpSize;

    using EncodeT = typename Codec::EncodeT;

    auto query = queries(queryId).data();
    auto vecsBase = (EncodeT*)allListData[listId];
    vecsBase = vecsBase + splitStart * d;

    constexpr auto kInit = Metric::kDirection ? kFloatMin : kFloatMax;

    __shared__ float smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];

    BlockSelect<
            float,
            int,
            Metric::kDirection,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kInit, -1, smemK, smemV, k);

    // The codec might be dependent upon data that we need to reference or store
    // in shared memory
    codec.initKernel(smem, dim);
    __syncthreads();

    // How many vector blocks of 32 are in this list?
    int numBlocks = utils::divUp(splitCnt, 32);

    // Number of EncodeT words per each dimension of block of 32 vecs
    constexpr int bytesPerVectorBlockDim = Codec::kEncodeBits * 32 / 8;
    constexpr int wordsPerVectorBlockDim =
            bytesPerVectorBlockDim / sizeof(EncodeT);
    int wordsPerVectorBlock = wordsPerVectorBlockDim * dim;

    int dimBlocks = utils::roundDown(dim, kWarpSize);

    for (int block = warpId; block < numBlocks; block += kNumWarps) {
        // We're handling a new vector
        Metric dist = metric.zero();

        // This is the vector a given lane/thread handles
        int vec = block * kWarpSize + laneId;
        bool valid = vec < splitCnt;

        // This is where this warp begins reading data
        EncodeT* data = vecsBase + block * wordsPerVectorBlock;

        // whole blocks
        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            int loadDim = dBase + laneId;
            float queryReg = query[loadDim];

            constexpr int kUnroll = 4;

#pragma unroll
            for (int i = 0; i < kWarpSize / kUnroll;
                 ++i, data += kUnroll * wordsPerVectorBlockDim) {
                EncodeT encV[kUnroll];
#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    encV[j] = WarpPackedBits<EncodeT, Codec::kEncodeBits>::read(
                            laneId, data + j * wordsPerVectorBlockDim);
                }

#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    encV[j] = WarpPackedBits<EncodeT, Codec::kEncodeBits>::
                            postRead(laneId, encV[j]);
                }

                float decV[kUnroll];
#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    int d = i * kUnroll + j;
                    decV[j] = codec.decodeNew(dBase + d, encV[j]);
                }


#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    int d = i * kUnroll + j;
                    float q = SHFL_SYNC(queryReg, d, kWarpSize);
                    dist.handle(q, decV[j]);
                }
            }
        }

        // remainder
        int loadDim = dimBlocks + laneId;
        bool loadDimInBounds = loadDim < dim;

        float queryReg = loadDimInBounds ? query[loadDim] : 0;

        for (int d = 0; d < dim - dimBlocks;
             ++d, data += wordsPerVectorBlockDim) {
            float q = SHFL_SYNC(queryReg, d, kWarpSize);

            EncodeT enc = WarpPackedBits<EncodeT, Codec::kEncodeBits>::read(
                    laneId, data);
            enc = WarpPackedBits<EncodeT, Codec::kEncodeBits>::postRead(
                    laneId, enc);
            float dec = codec.decodeNew(dimBlocks + d, enc);

            dist.handle(q, dec);
        }

        if (valid) {
            heap.addThreadQ(dist.reduce(), vec + splitStart);
        }

        heap.checkThreadQ();
    }

    heap.reduce();

    auto distanceOutBase = best_distances(queryno)[probeId].data();
    auto indicesOutBase =  best_indices(queryno)[probeId].data();
    distanceOutBase += splitId * k;
    indicesOutBase += splitId * k;

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        distanceOutBase[i] = smemK[i];
        indicesOutBase[i] = smemV[i];
    }
}






//
// We split up the scan function into multiple compilation units to cut down on
// compile time using these macros to define the function body
//

#define IVFINT_RUN_PIPE(CODEC_TYPE, METRIC_TYPE, THREADS, NUM_WARP_Q, NUM_THREAD_Q) \
    do {                                                                       \
        dim3 grid(maxcluster_per_query, nquery, split);                        \
        KernelCompute<                                                         \
                    CODEC_TYPE,                                                \
                    METRIC_TYPE,                                               \
                    THREADS,                                                   \
                    NUM_WARP_Q,                                                \
                    NUM_THREAD_Q,                                              \
                    true><<<grid, THREADS, codec.getSmemSize(d), stream>>>(    \
                    d,                                                         \
                    k,                                                         \
                    queryids,                                                   \
                    queries,                                                   \
                    query_cluster_matrix,                                     \
                    best_indices,                                              \
                    best_distances,                                            \
                    deviceListDataPointers_,                                   \
                    deviceListLengths_,                                        \
                    metric,                                                    \
                    codec);                                                    \
    } while (0);

#define IVFINT_CODECS_PIPE(METRIC_TYPE, THREADS, NUM_WARP_Q, NUM_THREAD_Q)      \
    do {                                                                        \
        using CodecT = CodecFloat;                                              \
        CodecT codec(d * sizeof(float));                                      \
        IVFINT_RUN_PIPE(CodecT, METRIC_TYPE, THREADS, NUM_WARP_Q, NUM_THREAD_Q);\
    } while (0)

#define IVFINT_METRICS_PIPE(THREADS, NUM_WARP_Q, NUM_THREAD_Q)                 \
    do {                                                                       \
                                                                               \
        if (metric == MetricType::METRIC_L2) {                                 \
            L2Distance metric;                                                 \
            IVFINT_CODECS_PIPE(L2Distance, THREADS, NUM_WARP_Q, NUM_THREAD_Q); \
        } else if (metric == MetricType::METRIC_INNER_PRODUCT) {               \
            IPDistance metric;                                                 \
            IVFINT_CODECS_PIPE(IPDistance, THREADS, NUM_WARP_Q, NUM_THREAD_Q); \
        } else {                                                               \
            FAISS_ASSERT(false);                                               \
        }                                                                      \
    } while (0)

// for the interleaved by 32 layout
    /* compute the k nearest neighbor, given every (query, bcluster)
     * @param d dim d
     * @param k k nearest neighbors
     * @param nquery the total number of queries.
     * @param maxcluster_per_query max number of queries per cluster.
     * @param queryids[nquery] id of query input.
     * @param queries[nquery][d] all the queries.
     * @param query_cluster_matrix[nquery][maxcluster_per_query] 
     *  the queryIds for each cluster. The pointer matrix is on GPU.
     * @param best_indices[query][maxcluster_per_query][k] GPU space to save the k best's indices.
     * @param best_distances[query][maxcluster_per_query][k] GPU space to save the k best's distance.
     * @param deviceListDataPointers_[bnlist]
     * @param indicesOptions
     * @param deviceListLengths_[bnlist]
     * @param metric
     * @param stream
     */
void runKernelCompute(
    int d,
    int k,
    int nquery,
    int maxcluster_per_query,
    PipeTensor<int, 1, true> queryids,
    PipeTensor<float, 2, true> queries,
    PipeTensor<int, 2, true> query_cluster_matrix,
    PipeTensor<int, 3, true> best_indices,
    PipeTensor<float, 3, true> best_distances,
    void** deviceListDataPointers_,
    IndicesOptions indicesOptions,
    int* deviceListLengths_,
    faiss::MetricType metric,
    cudaStream_t stream,
    int split);



// Second pass of IVF list scanning to perform final k-selection and look up the
// user indices
   /** compute the k nearest neighbor for each query.
     * @param maxcluster_cnt max number of cluster per query
     * @param best_indices[nq][maxcluster_per_query][k] the k best indices for every (query, cluster). The pointer matrix is on GPU.
     *  (query,cluster) compute.
     * @param best_distances[nq][maxcluster_per_query] the k best distances for every (query, cluster). The pointer matrix is on GPU.
     *  (query,cluster) compute.
     * @param query_bcluster_matrix[nq][maxcluster_per_query] the pointer matrix is on GPU.
     * The pointers are pointed to GPU.
     * @param k k nearest neighbor
     * @param ListIndices[bcluster_cnt] the indices data of each cluster.
     * @param dir
     * @param out_indices[nq] the pointer of space to solve the final k best's indices.
     *  The pointer matrix is on GPU. The pointers are pointed to GPU.
     * @param out_distances[nq] the pointer of space to solve the final k best's distance.
     *  The pointer matrix is on GPU. The pointers are pointed to GPU.
     */
void runKernelReduce(
        int maxcluster_cnt,
        PipeTensor<float, 3, true> best_distances,
        PipeTensor<int, 3, true> best_indices,
        PipeTensor<int, 2, true> query_bcluster_matrix,
        int k,
        void** listIndices,
        IndicesOptions indicesOptions,
        bool dir,
        PipeTensor<float, 2, true> out_distances,
        PipeTensor<int, 2, true> out_indices,
        cudaStream_t stream,
        int split);



// Wrapper
   /** compute the k nearest neighbor for each query.
     * @param
     * @param
     * @param
     * @param
     * @param maxcluster_cnt max number of cluster per query
     * @param best_indices[nq][maxcluster_per_query][k] the k best indices for every (query, cluster). The pointer matrix is on GPU.
     *  (query,cluster) compute.
     * @param best_distances[nq][maxcluster_per_query] the k best distances for every (query, cluster). The pointer matrix is on GPU.
     *  (query,cluster) compute.
     * @param query_bcluster_matrix[nq][maxcluster_per_query] the pointer matrix is on GPU.
     * The pointers are pointed to GPU.
     * @param k k nearest neighbor
     * @param ListIndices[bcluster_cnt] the indices data of each cluster.
     * @param dir
     * @param out_indices[nq] the pointer of space to solve the final k best's indices.
     *  The pointer matrix is on GPU. The pointers are pointed to GPU.
     * @param out_distances[nq] the pointer of space to solve the final k best's distance.
     *  The pointer matrix is on GPU. The pointers are pointed to GPU.
     */
void runKernelComputeReduce(
    int d,
    int k,
    int nquery,
    int maxcluster_per_query,
    PipeTensor<int, 1, true> queryids,
    PipeTensor<float, 2, true> queries,
    PipeTensor<int, 2, true> query_cluster_matrix,
    void** deviceListDataPointers_,
    IndicesOptions indicesOptions,
    int* deviceListLengths_,
    void** listIndices,
    faiss::MetricType metric,
    bool dir,
    PipeTensor<float, 2, true> out_distances,
    PipeTensor<int, 2, true> out_indices,
    PipeCluster* pc,
    PipeGpuResources* pipe_res,
    int device,
    int split);



// Second pass of IVF list scanning to perform final k-selection and look up the
// user indices
   /** compute the k nearest neighbor for each query.
     * @param cnt_per_query[nq] the maximum total number of results for a query
     * @param result_indices[nq][maxcnt_per_query] the poniter of the space to
     * save the k best indices for every (query, cluster). The pointer matrix is on GPU.
     *  The pointers are ponited to GPU.
     * @param result_distances[nq][maxcluster_per_query] the poniter of the space to
     * save the k best distances for every (query, cluster). The pointer matrix is on GPU.
     *  The pointers are ponited to GPU.
     * @param k k nearest neighbor
     * @param indiceOptions
     * @param dir
     * @param out_indices[nq] the pointer of space to solve the final k best's indices.
     *  The pointer matrix is on GPU. The pointers are pointed to GPU.
     * @param out_distances[nq] the pointer of space to solve the final k best's distance.
     *  The pointer matrix is on GPU. The pointers are pointed to GPU.
     * @param stream
     */
void runKernelMerge(
    PipeTensor<int, 1, true> cnt_per_query,
    PipeTensor<float*, 2, true> result_distances,
    PipeTensor<int*, 2, true> result_indices,
    int k,
    IndicesOptions indicesOptions,
    bool dir,
    PipeTensor<float, 2, true> out_distances,
    PipeTensor<int, 2, true> out_indices,
    cudaStream_t stream);






} // namespace gpu
} // namespace faiss
