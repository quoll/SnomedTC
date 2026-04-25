NVCC      := nvcc
CPP       := g++
NVCCFLAGS_COMMON := -O3 -std=c++20 -Xcompiler "-Wall -Wextra -fopenmp"
CPPFLAGS  := -O3 -std=c++20 -Wall -Wextra
LDFLAGS   := -lgomp


NVCCFLAGS_ARCH := \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90

NVCCFLAGS := $(NVCCFLAGS_COMMON) $(NVCCFLAGS_ARCH)

SRCS   := snomed_tc.cu algorithm_a.cu algorithm_b.cu resulttx.cu doubling.cu
OBJS   := $(SRCS:.cu=.o)

.PHONY: all clean

all: snomed_tc algorithm_a algorithm_b algorithm_c

snomed_tc: snomed_tc.o resulttx.o doubling.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

algorithm_a: algorithm_a.o resulttx.o doubling.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

algorithm_b: algorithm_b.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

algorithm_c: algorithm_c.o
	$(CPP) -o $@ $^

algorithm_c.o: algorithm_c.cpp
	$(CPP) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

