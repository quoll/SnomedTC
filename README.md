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
Due to the size of the program, a truncated form of algorithm A is implemented, where the conversion of the final adjacency matrix
to source/destination pairs is performed using OpenMP on the host system.

This program requires the Nvidia CUDA compiler, and OpenMP.

### doubling
This program implements the full algorithm A. Final conversion from the adjacency matrix to source/destination pairs is performed on the GPU.

This program requires the Nvidia CUDA compiler.

### iterative
This program implements the full algorithm B.

This program requires the Nvidia CUDA compiler.

### serial
This program implements algorithm C.

Only the GNU C++ compiler is required.

## Building
To build everything, type:
```bash
make
```
To compile only a single program, name that program in the `make` command. e.g.
```bash
make serial
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
