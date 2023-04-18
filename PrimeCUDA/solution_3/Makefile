CUDA_DIR = /usr/local/cuda
GPU_ARCH = compute_75
GPU_CODE = sm_75

NVCC=$(CUDA_DIR)/bin/nvcc
NVCC_FLAGS=-ccbin g++ -std=c++14 -arch=$(GPU_ARCH) -code=$(GPU_CODE) -Xcompiler -march=native,-Ofast -lineinfo -lm -lstdc++ -I $(CUDA_DIR)/include/

all: primes

%: %.cu %.h
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm primes
