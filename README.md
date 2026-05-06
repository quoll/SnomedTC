# SNOMED Transitive Closure

The project implements 2 different algorithms to calculate the transitive closure of `isA` relationships in SNOMED-CT.
These relationships are calculated from the sct2_Relationship_Snapshot_INT_yyyyMMdd.txt file distributed in both the US and International publications of SNOMED-CT.

An example of this file from September 2025 is included.

## Algorithms
This project employs 2 different algorithms. Algorithm "A" performs a "join" operation across the set of all edges with itself.
This doubles the length of all the paths described in the initial set. The join is repeated until a fixed point is reached.
The data is represented as an adjacency matrix, and the join operations are performed on the GPU.

Algorithm "B" performs a similar join, but each step joins against the original adjacency list to extend paths by one hop.
This approach requires more iterations, but each step is significantly cheaper. Like Algorithm A, the data is represented
as an adjacency matrix, with join operations executed on the GPU.

Algorithm "C" is a repetition of Algorithm A, but performed serially on a CPU,
using HashMap and HashSet data structures to avoid the memory requirements of an adjacency matrix.
This version is provided for serialized comparison.

## Requirements
This project includes several programs, with each applying different approaches. They all need different libraries and compilers.
All programs make use of features from C++ 20.

### snomed_ct
This program implements algorithms A, B, and C.
It compares and contrasts the timing and results of each step for each algorithm.

This program requires the Nvidia CUDA compiler.

### algorithm_a
This program implements the full algorithm A.

This program requires the [Nvidia CUDA compiler](https://developer.nvidia.com/cuda-downloads): NVCC.

### algorithm_b
This program implements the full algorithm B.

This program requires the [Nvidia CUDA compiler](https://developer.nvidia.com/cuda-downloads): NVCC.

### algorithm_c
This program implements algorithm C. This is similar to algorithm A, but performed serially on a CPU, using a hashmap to hashsets in place of the sparse array.

Only the [GNU C++ compiler](https://gcc.gnu.org/) is required.

## Building
To build everything, type:
```bash
make
```
To compile only a single program, name that program in the `make` command. e.g.
```bash
make algorithm_c
```
Individual programs should be buildable on systems that cannot compile the other programs.
For instance, `make serial` should work on MacOS, despite this platform not supporting CUDA.

## Execution
To run a program, provide arguments of a SNOMED-CT RelationshipSnapshot file, and an output filename.
```bash
./snomed_ct sct2_Relationship_Snapshot_INT_20250901.txt output.txt
```

## SNOMED-CT and The Unified Medical Language System (UMLS)
UMLS Knowledge Sources \[dataset on the Internet\]. Release 2024AA.
Bethesda (MD): National Library of Medicine (US); 2024 May 6 \[cited 2024 Jul 15\].
Available from: [http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html](http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)

## File Descriptions
* `algorithm_a.cu`: `main()` function for the `algorithm_a` program. Depends on `graph_util.h`/`graph_util.cpp`, `doubling.cuh`/`doubling.cu`, and `resulttx.cuh`/`resulttx.cu`.
* `algorithm_b.cu`: `main()` function for the `algorithm_b` program. Depends on `graph_util.h`/`graph_util.cpp`, `iterative.cuh`/`iterative.cu`, and `resulttx.cuh`/`resulttx.cu`.
* `algorithm_c.cpp`: `main()` function for the `algorithm_b` program. Depends on `graph_util.h`/`graph_util.cpp`, and `serial.h`/`serial.cpp`.
* `common.cuh`: header file containing data structures and inline functions for CUDA representation and transfer.
* `doubling.cu`: implements the steps for algorithm A on CUDA, setting up the graph matrix and joining against itself.
* `doubling.cuh`: header for the algorithm A operations.
* `graph_util.cpp`: implements functions for reading SNOMED-CT, and setting up the graph data structures.
* `graph_util.h`: header for reading SNOMED-CT, and setting up the graph data structures.
* `iterative.cu`: implements the steps for algorithm B on CUDA, setting up the graph matrix and joining against a CSR representation.
* `iterative.cuh`: header for the algorithm B operations.
* `Makefile`: build operations to generate programs. All targets except `algorithm_c` require the `nvcc` compiler from NVIDIA.
* `relationship-data.zip`: compressed relationship file from SNOMED-CT, International Edition, September 2025.
* `resulttx.cu`: implements functions for converting a final result into a CSR-style structure on a GPU, transferring to the host, and converting to graph edges.
* `resulttx.cuh`: header for the result-transfer operations in `resulttx.cu`.
* `serial.cpp`: implements the steps for the graph self-join algorithm serially on a CPU, using a hashmap/hashset structure for the graph.
* `serial.h`: header for the `serial.cpp` functions.
* `snomed_tc.cu`: `main()` function for the `snomed_tc` program. This is a combined form of `algorithm_a`/`algorithm_b`/`algorithm_c`, and compares the results of each for equality.

## License
MIT License
