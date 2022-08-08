/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cassert>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>
#include <omp.h>
#include <cinttypes>
#include <stdint.h>
#include <algorithm>
#include <mutex>
#include <string.h>
#include <limits>
#include <memory>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/pipe/IndexIVFPipe.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/PipeTensor.cuh>
#include <faiss/pipe/PipeScheduler.h>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/pipe/PipeKernel.cuh>
#include <thrust/device_vector.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

bool file_exist(const std::string& file_path)
{
	if (FILE* file = fopen(file_path.c_str(), "r")){
		fclose(file);
		return true;
	}
	else 
		return false;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

int main(){
    omp_set_num_threads(8);
    auto t0 = elapsed();

    int dim = 128;
    int dev_no = 0;
    int ncentroids = 64;
    faiss::gpu::StandardGpuResources resources;

    faiss::gpu::PipeGpuResources* pipe_res = new faiss::gpu::PipeGpuResources();
    faiss::IndexIVFPipeConfig config;
    faiss::IndexIVFPipe* index = new faiss::IndexIVFPipe(dim, ncentroids, config, pipe_res, faiss::METRIC_L2);

    FAISS_ASSERT (config.interleavedLayout == true);

    size_t d;
    // Train the index
    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read("/workspace/data/sift/sift10M/sift10M.fvecs", &d, &nt);

        FAISS_ASSERT(d == dim);

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete[] xt;
    }

    // Add the data
    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float* xb = fvecs_read("/workspace/data/sift/sift10M/sift10M.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);

        index->add(nb, xb);

        delete[] xb;
    }

    size_t nq;
    float* xq;
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("/workspace/data/sift/sift10M/query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;                // nb of results per query in the GT
    faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read("/workspace/data/sift/sift10M/idx.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    float *gtd;
    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        gtd = fvecs_read("/workspace/data/sift/sift10M/dis.fvecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
    }
    printf("[%.3f s] Start Balancing\n",
               elapsed() - t0);
    index->balance();
    printf("[%.3f s] Finishing Balancing: %d B clusters\n",
               elapsed() - t0, index->pipe_cluster->bnlist);

    auto pc = index->pipe_cluster;
    pipe_res->initializeForDevice(0, pc);

    printf("[%.3f s] Start Profile\n",
               elapsed() - t0);
    // Train profile
    if (!file_exist("./profiler.txt")){
        index->profile();
        index->saveProfile("./profiler.txt");
    }
    else{
        index->loadProfile("./profiler.txt");
    }
    printf("[%.3f s] Finish Profile\n",
               elapsed() - t0);

    int bs = 256;
    int topk = 10;
    std::vector<float> dis(bs * topk);
    std::vector<int> idx(bs * topk);
    index->set_nprobe(ncentroids / 2);

    auto tt0 = elapsed();

    auto sche = new faiss::gpu::PipeScheduler(index, 
            pc, pipe_res, bs, xq, topk, dis.data(), idx.data());
    auto tt1 = elapsed();
    printf("Search Time: %.3f ms\n", (tt1 - tt0)*1000);
    delete sche;

    for (int i = 0; i < topk; i++){
        printf("%d %d: %f %f\n", idx[i + topk * 128], gt[i + 100 * 128], dis[i + topk * 128], gtd[i + 100 * 128]);
    }

    tt0 = elapsed();

    sche = new faiss::gpu::PipeScheduler(index, 
            pc, pipe_res, bs, xq + d * bs, topk, dis.data(), idx.data());
    tt1 = elapsed();
    printf("Second Search Time: %.3f ms\n", (tt1 - tt0)*1000);
    
    delete sche;

    for (int i = 0; i < topk; i++){
        printf("%d %d: %f %f\n", idx[i], gt[i + 100 * bs], dis[i], gtd[i + 100 * bs]);
    }


    delete[] xq;
    delete[] gt;
    delete[] gtd;
    delete index;
    return 0;
}