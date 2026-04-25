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

CU_SRCS  := snomed_tc.cu algorithm_a.cu algorithm_b.cu resulttx.cu doubling.cu iterative.cu
CPP_SRCS := algorithm_c.cpp serial.cpp graph_util.cpp
CU_OBJS  := $(CU_SRCS:.cu=.o)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)

.PHONY: all clean

all: snomed_tc algorithm_a algorithm_b algorithm_c

snomed_tc: snomed_tc.o resulttx.o doubling.o iterative.o serial.o graph_util.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

algorithm_a: algorithm_a.o resulttx.o doubling.o graph_util.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

algorithm_b: algorithm_b.o resulttx.o iterative.o graph_util.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

algorithm_c: algorithm_c.o serial.o graph_util.o
	$(CPP) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) snomed_tc algorithm_a algorithm_b algorithm_c
