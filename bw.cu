#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <thread>
#include <atomic>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

std::atomic<int> finish;
std::atomic<int> barrier;

void testBandwidth(int gpu_id, size_t dataSize, int numTransfers, int cnt, int worker_id) {
    char *hostData, *deviceData;
    cudaStream_t stream;
    cudaEvent_t start, stop;

    gpuErrchk(cudaSetDevice(gpu_id));
    gpuErrchk(cudaMallocHost(&hostData, dataSize)); // Allocate pinned host memory
    gpuErrchk(cudaMalloc(&deviceData, dataSize));
    gpuErrchk(cudaStreamCreate(&stream));
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    float totalMilliseconds = 0;

    barrier++;
    while (barrier < cnt);

    for (int i = 0; i < numTransfers; ++i) {
        gpuErrchk(cudaEventRecord(start, stream));
        gpuErrchk(cudaMemcpyAsync(deviceData, hostData, dataSize, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaEventRecord(stop, stream));
        gpuErrchk(cudaEventSynchronize(stop));

        float milliseconds = 0;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
        totalMilliseconds += milliseconds;
    }

    barrier--;
    while (barrier > 0);

    float averageMilliseconds = totalMilliseconds / numTransfers;
    float averageBandwidth = dataSize / (averageMilliseconds * 1e6); // GB/s

    while (finish != worker_id);
    std::cout << "GPU " << gpu_id << " Average Bandwidth: " << averageBandwidth << " GB/s" << std::endl;
    finish ++;
    // Cleanup
    cudaFree(deviceData);
    cudaFreeHost(hostData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <GPU_ID1> [<GPU_ID2> ...]" << std::endl;
        return -1;
    }

    const int numTransfers = 1000; // Number of transfers to average
    std::vector<std::thread> threads;
    const size_t dataSize = 1024 * 1024 * 256; // 256 MB
    finish = 0;

    for (int i = 1; i < argc; ++i) {
        int gpu_id = std::atoi(argv[i]);
        threads.emplace_back(testBandwidth, gpu_id, dataSize, numTransfers, argc-1, i-1);
    }

    // Join all threads
    for (auto &th : threads) {
        th.join();
    }

    return 0;
}
