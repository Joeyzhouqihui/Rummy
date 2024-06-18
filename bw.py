import torch 
import time
import threading
import sys

def work(gpu_id):
    num = 1000000000
    data = torch.ones(1, 1000000000, dtype=torch.float32, pin_memory=True)
    num_iter = 20
    time_cost = 0
    for i in range(num_iter):
        time_cost -= time.time()
        data_gpu = data.to(gpu_id)
        time_cost += time.time()
    print(gpu_id, " bandwidth: ", (4 * num_iter / time_cost), " GB/s")

if __name__ == "__main__":
    work(int(sys.argv[1]))