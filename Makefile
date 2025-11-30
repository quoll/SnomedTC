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

SRCS   := snomed_tc.cu doubling.cu iterative.cu
OBJS   := $(SRCS:.cu=.o)

.PHONY: all clean

all: snomed_ct doubling iterative serial

snomed_tc: snomed_tc.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

doubling: doubling.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

iterative: iterative.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

serial: serial.o
	$(CPP) -o $@ $^

serial.o: serial.cpp
	$(CPP) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

