CXX      := /usr/bin/clang++

# Linux flags
# CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -pedantic -fopenmp

# MacOS flags for with explicit paths for OMP
CXXFLAGS := -Xpreprocessor -fopenmp -O3 -std=c++20 -Wall -Wextra -pedantic -I$(shell brew --prefix)/include -I/opt/homebrew/opt/libomp/include
LDFLAGS  := -lomp -L/opt/homebrew/opt/libomp/lib

TARGET   := snomed_tc
SRCS     := main.cpp index_timing.cpp csr_index_timing.cpp
OBJS     := $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

index_timing: index_timing.o
	$(CXX) -o $@ $^ $(LDFLAGS)

csr_index_timing: csr_index_timing.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) index_timing csr_index_timing

