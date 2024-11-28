#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <immintrin.h> // For AVX
#include <algorithm>
#include <cassert>

using namespace std;
using namespace std::chrono;

// Function to initialize a random n x n matrix
void initializeMatrix(vector<vector<float>> &matrix, int n) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// Function to make the matrix symmetric
void makeSymmetric(vector<vector<float>> &matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            matrix[i][j] = matrix[j][i];
        }
    }
}

// Sequential symmetry check
bool checkSym(const vector<vector<float>> &matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (matrix[i][j] != matrix[j][i]) {
                return false;
            }
        }
    }
    return true;
}

bool checkSymImp(const vector<vector<float>> &matrix, int n) {
    const int smallMatrixThreshold = 64;
    bool isSymmetric = true;

    if (n <= smallMatrixThreshold) {
        // Manual Loop Unrolling with Prefetching for Small Matrices
        for (int i = 0; i < n && isSymmetric; ++i) {
            // Prefetch the upcoming row for cache locality
            if (i + 1 < n) {
                __builtin_prefetch(&matrix[i + 1][0], 0, 1);
            }
            for (int j = i + 1; j < n; j += 8) { // Process 8 elements at a time
                if (j < n && matrix[i][j] != matrix[j][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 1 < n && matrix[i][j + 1] != matrix[j + 1][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 2 < n && matrix[i][j + 2] != matrix[j + 2][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 3 < n && matrix[i][j + 3] != matrix[j + 3][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 4 < n && matrix[i][j + 4] != matrix[j + 4][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 5 < n && matrix[i][j + 5] != matrix[j + 5][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 6 < n && matrix[i][j + 6] != matrix[j + 6][i]) {
                    isSymmetric = false;
                    break;
                }
                if (j + 7 < n && matrix[i][j + 7] != matrix[j + 7][i]) {
                    isSymmetric = false;
                    break;
                }

                // Prefetch upcoming elements for next iteration to improve memory access patterns
                if (j + 8 < n) {
                    __builtin_prefetch(&matrix[i][j + 8], 0, 1);
                    __builtin_prefetch(&matrix[j + 8][i], 0, 1);
                }
            }
        }
    } else {
        // AVX-512 implementation for larger matrices
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; j += 16) {
                // Remaining elements less than 16
                if (j + 16 > n) {
                    for (int k = j; k < n; ++k) {
                        if (matrix[i][k] != matrix[k][i]) {
                            return false;
                        }
                    }
                    break;
                }
                
                // Load 16 float values from matrix[i] and matrix[j]
                __m512 row_i = _mm512_loadu_ps(&matrix[i][j]);
                __m512 col_j = _mm512_set_ps(
                    matrix[j][j+15], matrix[j][j+14], matrix[j][j+13], matrix[j][j+12],
                    matrix[j][j+11], matrix[j][j+10], matrix[j][j+9], matrix[j][j+8],
                    matrix[j][j+7], matrix[j][j+6], matrix[j][j+5], matrix[j][j+4],
                    matrix[j][j+3], matrix[j][j+2], matrix[j][j+1], matrix[j][j]
                );
                
                // Compare for inequality
                __mmask16 cmp_mask = _mm512_cmpneq_ps_mask(row_i, col_j);
                
                if (cmp_mask != 0) {
                    return false;
                }
            }
        }
    }

    return isSymmetric;
}

// OpenMP parallelized symmetry check
bool checkSymOMP(const vector<vector<float>> &matrix, int n) {
    bool isSymmetric = true;
	const int smallMatrixThreshold = 64;
	const int blockSize = 64;

    if (n <= smallMatrixThreshold) {
		#pragma omp parallel for shared(isSymmetric)
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				// Early exit if symmetry is already violated
				if (!isSymmetric) continue;

				if (matrix[i][j] != matrix[j][i]) {
					#pragma omp atomic write
					isSymmetric = false;

					#pragma omp flush(isSymmetric)  // Ensure visibility of change
				}
			}
		}
	}
	else {
		#pragma omp parallel for shared(isSymmetric)
		for (int bi = 0; bi < n; bi += blockSize) {
			for (int bj = bi; bj < n; bj += blockSize) {
				for (int i = bi; i < min(bi + blockSize, n); ++i) {
					for (int j = max(i + 1, bj); j < min(bj + blockSize, n); ++j) {
						// Early exit if symmetry is violated
						if (!isSymmetric) continue;

						if (matrix[i][j] != matrix[j][i]) {
							#pragma omp atomic write
							isSymmetric = false;

							#pragma omp flush(isSymmetric)  // Ensure visibility of change
						}
					}
				}
			}
		}
	}
    return isSymmetric;
}

// Sequential transpose function
void matTranspose(const vector<vector<float>> &matrix, vector<vector<float>> &transpose, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transpose[j][i] = matrix[i][j];
        }
    }
}

// Implicit parallelized transpose
void matTransposeImp(const vector<vector<float>> &matrix, vector<vector<float>> &transpose, int n) {
    const int smallMatrixThreshold = 64;
	const int blockSize = 64;
    if (n <= smallMatrixThreshold) {
        // Manual Loop Unrolling with Prefetching for Small Matrices
        for (int i = 0; i < n; ++i) {
            if (i + 1 < n) {
                __builtin_prefetch(&matrix[i + 1][0], 0, 1);
            }
            for (int j = 0; j < n; j += 8) { // Process 8 elements at a time
                if (j < n) transpose[j][i] = matrix[i][j];
                if (j + 1 < n) transpose[j + 1][i] = matrix[i][j + 1];
                if (j + 2 < n) transpose[j + 2][i] = matrix[i][j + 2];
                if (j + 3 < n) transpose[j + 3][i] = matrix[i][j + 3];
                if (j + 4 < n) transpose[j + 4][i] = matrix[i][j + 4];
                if (j + 5 < n) transpose[j + 5][i] = matrix[i][j + 5];
                if (j + 6 < n) transpose[j + 6][i] = matrix[i][j + 6];
                if (j + 7 < n) transpose[j + 7][i] = matrix[i][j + 7];

                if (j + 8 < n) {
                    __builtin_prefetch(&matrix[i][j + 8], 0, 1);
                    __builtin_prefetch(&transpose[j + 8][i], 1, 1);
                }
            }
        }
    } else {
        // Vectorized Processing with AVX2 for Large Matrices
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; j += 16) {
				if (j + 16 < n) {
					// Load 16 floats from the current row of the matrix
					__m512 row = _mm512_loadu_ps(&matrix[i][j]);
					// Store them transposed into the transpose matrix
					for (int k = 0; k < 16; ++k) {
						transpose[j + k][i] = matrix[i][j + k]; // can not use the function _mm512_storeu_ps because it requires contiguous memory which isn't possible in 2D vectors
					}
				} else {
					// Handle remaining elements that don't fit into 16-wide chunks
					for (int k = j; k < n; ++k) {
						transpose[k][i] = matrix[i][k];
					}
				}
			}
		}
    }
}

// OpenMP parallelized transpose with block-based optimization
void matTransposeOMP(const vector<vector<float>> &matrix, vector<vector<float>> &transpose, int n) {
    const int smallMatrixThreshold = 64;
    const int blockSize = 64;

    if (n <= smallMatrixThreshold) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                transpose[j][i] = matrix[i][j];
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i += blockSize) {
            for (int j = 0; j < n; j += blockSize) {
                for (int ii = i; ii < min(i + blockSize, n); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, n); ++jj) {
                        transpose[jj][ii] = matrix[ii][jj];
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);

    // Validate matrix size and ensure n is a power of 2
    if (n <= 0 || (n & (n - 1)) != 0) {
        cerr << "Matrix size must be a positive power of 2" << endl;
        return 1;
    }

    // Initialize matrix
    vector<vector<float>> M(n, vector<float>(n));
    initializeMatrix(M, n);

    // Measure execution times
    vector<vector<float>> T(n, vector<float>(n));
    
    // Sequential symmetry check
    auto startSymCheck = high_resolution_clock::now();
    bool isSymmetric = checkSym(M, n);
    auto endSymCheck = high_resolution_clock::now();
    auto durationSymCheck = duration_cast<microseconds>(endSymCheck - startSymCheck);

    // Implicit parallel symmetry check
    auto startSymCheckImp = high_resolution_clock::now();
    bool isSymmetricImp = checkSymImp(M, n);
    auto endSymCheckImp = high_resolution_clock::now();
    auto durationSymCheckImp = duration_cast<microseconds>(endSymCheckImp - startSymCheckImp);

    // OpenMP symmetry check
    auto startSymCheckOMP = high_resolution_clock::now();
    bool isSymmetricOMP = checkSymOMP(M, n);
    auto endSymCheckOMP = high_resolution_clock::now();
    auto durationSymCheckOMP = duration_cast<microseconds>(endSymCheckOMP - startSymCheckOMP);

    // Sequential transpose
    auto startTranspose = high_resolution_clock::now();
    matTranspose(M, T, n);
    auto endTranspose = high_resolution_clock::now();
    auto durationTranspose = duration_cast<microseconds>(endTranspose - startTranspose);

    // Implicit parallel transpose
    auto startTransposeImp = high_resolution_clock::now();
    matTransposeImp(M, T, n);
    auto endTransposeImp = high_resolution_clock::now();
    auto durationTransposeImp = duration_cast<microseconds>(endTransposeImp - startTransposeImp);

    // OpenMP parallel transpose
    auto startTransposeOMP = high_resolution_clock::now();
    matTransposeOMP(M, T, n);
    auto endTransposeOMP = high_resolution_clock::now();
    auto durationTransposeOMP = duration_cast<microseconds>(endTransposeOMP - startTransposeOMP);

    // Output results
    cout << "Sequential Symmetry Check Time: " << durationSymCheck.count() << " microseconds\n";
    cout << "Implicit Symmetry Check Time: " << durationSymCheckImp.count() << " microseconds\n";
    cout << "OpenMP Symmetry Check Time: " << durationSymCheckOMP.count() << " microseconds\n";
    cout << "Is matrix symmetric (Sequential)? " << (isSymmetric ? "Yes" : "No") << endl;
    cout << "Is matrix symmetric (Implicit)? " << (isSymmetricImp ? "Yes" : "No") << endl;
    cout << "Is matrix symmetric (OpenMP)? " << (isSymmetricOMP ? "Yes" : "No") << endl;
    cout << "Sequential Transpose Time: " << durationTranspose.count() << " microseconds\n";
    cout << "Implicit Transpose Time: " << durationTransposeImp.count() << " microseconds\n";
    cout << "OpenMP Transpose Time: " << durationTransposeOMP.count() << " microseconds\n";

    return 0;
}
