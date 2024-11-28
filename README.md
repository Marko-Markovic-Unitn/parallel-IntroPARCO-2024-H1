# Matrix Transpose and Symmetry Optimization

This repository contains code for optimizing matrix transposition and symmetry operations using both sequential and parallel approaches. The implementation leverages OpenMP for parallelism and employs techniques like block-wise partitioning, and vectorized operations (AVX-512) for high performance.

---

### Prerequisites

Ensure you have the following tools and libraries installed:

- **Compiler**: GCC or Clang (with OpenMP support).
- **AVX-512**: Compatible hardware for vectorized operations

### Installation

1. Clone the repository
2. Compile the program with: g++ -O2 -fopenmp -march=native -ftree-vectorize -fprefetch-loop-arrays -o matrix_parallel deliverable.c

### How to Run
1. Modify, if needed, run.pbs by putting the number of threads and/or changing other parameters like the matrix size
2. Run with qsub run.pbs
3. Alternatively run the program manually with ./matrix_parallel [matrix_size]