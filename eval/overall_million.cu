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

#include <faiss/pipe/IndexIVFPipe.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/gpu/utils/PipeTensor.cuh>
#include <faiss/pipe/PipeScheduler.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/pipe/PipeKernel.cuh>

#include <torch/script.h>
#include <fstream>
#include <ios>

float* load_dataset(std::string filename, int &n, int &d) {
    FILE* f = fopen(filename.c_str(), "r");
    fread(&n, 1, sizeof(int), f);
    fread(&d, 1, sizeof(int), f);
    // printf("dataset num: %d, dim: %d\n", n, d);
    // torch::Tensor embeddings = torch::empty({n, d}, torch::TensorOptions().dtype(torch::kFloat32));
    // size_t nr = fread(embeddings.data_ptr<float>(), sizeof(float), size_t(n) * size_t(d), f);
    float *embeddings = new float[size_t(n) * size_t(d)];
    fread(embeddings, sizeof(float), size_t(n) * size_t(d), f);
    fclose(f);
    return embeddings;
}

void add_dataset_chunk(std::string filename, int n, int d, faiss::IndexIVFPipe* index) {
    FILE* f = fopen(filename.c_str(), "r");
    int n1, d1;
    fread(&n1, 1, sizeof(int), f);
    fread(&d1, 1, sizeof(int), f);
    assert(n1 == n);
    assert(d1 == d);
    int buffer_size = 1000000;
    float *buffer = new float[size_t(buffer_size) * size_t(d)];
    int pos = 0, num;
    while (pos < n) {
        if (pos + buffer_size >= n) {
            num = n - pos;
        } else {
            num = buffer_size;
        }
        fread(buffer, sizeof(float), size_t(num) * size_t(d), f);
        index->add(num, buffer);
        pos += buffer_size;
    }
    delete[] buffer;
    fclose(f);
}

void warmup(
        faiss::IndexIVFPipe* index, 
        faiss::gpu::PipeGpuResources* pipe_res,
        faiss::PipeCluster *pc,
        float *queries, int nq, int d, 
        int bs, int top_k,
        std::vector<float> &dis, std::vector<int> &idx) {
    int batch_cnt = 0;
    for (int i = 0; i < nq / bs; i++) {
        auto sche = new faiss::gpu::PipeScheduler(
            index, 
            pc, pipe_res, bs,
            queries + d * bs * i, 
            top_k, 
            dis.data() + top_k * bs * i, 
            idx.data() + top_k * bs * i);
        delete sche;
        batch_cnt++;
        if (batch_cnt >= 1000) {
            break;
        }
    }
}

void test_queries(
        faiss::IndexIVFPipe* index, 
        faiss::gpu::PipeGpuResources* pipe_res,
        faiss::PipeCluster *pc,
        float *queries, int nq, int d, 
        int bs, int top_k,
        std::vector<float> &dis, std::vector<int> &idx) {
    double total_batch_time = 0;
    double total_sample_time = 0;
    double total_group_time = 0;
    double total_reorder_time = 0;
    double total_computation_time = 0;
    double total_communication_time = 0;
    int batch_cnt = 0;
    for (int i = 0; i < nq / bs; i++) {
        double batch_time = omp_get_wtime();
        auto sche = new faiss::gpu::PipeScheduler(
            index, 
            pc, pipe_res, bs,
            queries + d * bs * i, 
            top_k, 
            dis.data() + top_k * bs * i, 
            idx.data() + top_k * bs * i);
        batch_time = omp_get_wtime() - batch_time;
        total_batch_time += batch_time;
        total_sample_time += sche->sample_time;
        total_group_time += sche->group_time;
        total_reorder_time += sche->reorder_time;
        total_computation_time += sche->com_time;
        total_communication_time += sche->com_transmission;
        delete sche;
        batch_cnt++;
        if (batch_cnt >= 100) {
            break;
        }
    }
    total_batch_time = total_batch_time * 1000 / batch_cnt;
    total_sample_time = total_sample_time * 1000 / batch_cnt;
    total_group_time = total_group_time * 1000 / batch_cnt;
    total_reorder_time = total_reorder_time * 1000 / batch_cnt;
    total_computation_time = total_computation_time * 1000 / batch_cnt;
    total_communication_time = total_communication_time * 1000 / batch_cnt;
    std::ofstream result_file("result", std::ios::app);
    result_file<<bs<<" ";
    // result_file<<total_sample_time<<" ";
    // result_file<<total_group_time<<" ";
    // result_file<<total_reorder_time<<" ";
    result_file<<total_computation_time<<" ";
    result_file<<total_communication_time<<" ";
    result_file<<total_batch_time<<std::endl;
}

// ./script dataset-name bs topk nprobe (./overall deep 256 10 1)
int main(int argc,char **argv) {
    std::string dataset = argv[1];
    int top_k = atoi(argv[2]);
    std::string db_file, index_file, query_file, dtype;
    faiss::MetricType metric_type;
    if (dataset == "turing") {
        db_file = "/data/MSTuringANNS/base1b.fbin.sample_nb_100000000";
        index_file = "/data/MSTuringANNS/base.100M.fbin.index";
        query_file = "/data/MSTuringANNS/query100K.fbin.sample";
        dtype = "float32";
        metric_type = faiss::MetricType::METRIC_L2;
    } else if (dataset == "specev") {
        db_file = "/home/ubuntu/projects/big-ann-benchmarks/data/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_100000000";
        index_file = "/home/ubuntu/projects/big-ann-benchmarks/data/MSSPACEV1B/base.100M.fbin.index";
        query_file = "/home/ubuntu/projects/big-ann-benchmarks/data/MSSPACEV1B/private_query_30k.bin";
        dtype = "int8";
        metric_type = faiss::MetricType::METRIC_L2;
    } else if (dataset == "wiki") {
        db_file = "/home/ubuntu/projects/big-ann-benchmarks/data/wiki/wikipedia.fbin.crop_nb_100000000";
        index_file = "/home/ubuntu/projects/big-ann-benchmarks/data/wiki/base.fbin.index";
        query_file = "/home/ubuntu/projects/big-ann-benchmarks/data/wiki/google_nq.bin";
        dtype = "float32";
        metric_type = faiss::MetricType::METRIC_L2;
    } else {
        printf("Your input dataset is not included yet! \n");
        return 0;
    }
    // build index
    printf("building index...\n");
    omp_set_num_threads(40);
    faiss::gpu::PipeGpuResources* pipe_res = new faiss::gpu::PipeGpuResources();
    faiss::IndexIVFPipeConfig config;
    faiss::IndexIVFPipe* index;
    int ncentroids, nb, d;
    {//train
        float* database = load_dataset(db_file, nb, d);
        std::cout<<"database: "<<nb<<" "<<d<<std::endl;
        ncentroids = int(sqrt(nb));
        index = new faiss::IndexIVFPipe(d, ncentroids, config, pipe_res, metric_type);
        FAISS_ASSERT (config.interleavedLayout == true);
        index->train(nb, database);
        delete[] database;
        add_dataset_chunk(db_file, nb, d, index);
    }
    // load queries
    printf("loading queries...\n");
    int nq, dq;
    float* queries = load_dataset(query_file, nq, dq);
    assert(d == dq);
    std::cout<<"queries: "<<nq<<" "<<dq<<std::endl;
    // balance
    printf("start balancing...\n");
    index->balance();
    omp_set_num_threads(8);
    printf("Finishing Balancing: %d B clusters\n", index->pipe_cluster->bnlist);
    faiss::PipeCluster *pc = index->pipe_cluster;
    // 设置内存分配的page大小，分配temp内存和cache内存，初始化cache的avl-tree
    pipe_res->initializeForDevice(0, pc);
    // Profile
    printf("start profiling...\n");
    index->profile();
    // Start queries
    std::vector<int> bs_list = {8, 16, 32, 64};
    std::vector<float> ratio_list = {0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05};
    std::vector<float> dis(nq * top_k);
    std::vector<int> idx(nq * top_k);
    index->set_nprobe(int(0.01 * ncentroids));
    warmup(
        index, pipe_res, pc,
        queries, nq, d,
        8, top_k,
        dis, idx
    );
    for (auto bs : bs_list) {
        for (auto ratio : ratio_list) {
            int n_probe = int(ratio * ncentroids);
            index->set_nprobe(n_probe);
            test_queries(
                index, pipe_res, pc,
                queries, nq, d,
                bs, top_k,
                dis, idx
            );
        }
    }
    delete index;
    delete pipe_res;
    delete[] queries;
    return 0;
}
