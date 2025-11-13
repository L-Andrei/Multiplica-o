#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>
#include <random>
#include <iostream>
#include <algorithm>
#include <immintrin.h>

using namespace ifnum::linearAlgebra;

//g++ -O3 -march=native -mfma -mavx2 matrix_mul.cpp -o matrix_mul

// Multiplicação de matrizes com blocagem hierárquica + SIMD
template<typename T>
void multiply_blocked(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C,
                      int blockSize, int subBlockSize) {
    const int n = A.rows();
    const int m = B.cols();
    const int p = A.cols();

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int kk = 0; kk < p; kk += blockSize) {

                int i_max = std::min(ii + blockSize, n);
                int j_max = std::min(jj + blockSize, m);
                int k_max = std::min(kk + blockSize, p);

                for (int iii = ii; iii < i_max; iii += subBlockSize) {
                    for (int jjj = jj; jjj < j_max; jjj += subBlockSize) {
                        for (int kkk = kk; kkk < k_max; kkk += subBlockSize) {

                            int i_sub_max = std::min(iii + subBlockSize, i_max);
                            int j_sub_max = std::min(jjj + subBlockSize, j_max);
                            int k_sub_max = std::min(kkk + subBlockSize, k_max);

                            for (int i = iii; i < i_sub_max; i++) {
                                for (int k = kkk; k < k_sub_max; k++) {
                                    T a_ik = A(i, k);

                                    // Versão vetorizada (AVX2)
                                    int j = jjj;
                                    __m256d a_vec = _mm256_set1_pd(a_ik); // replica a_ik em 4 posições

                                    for (; j + 4 <= j_sub_max; j += 4) {
                                        __m256d b_vec = _mm256_loadu_pd(&B(k, j));   // carrega 4 valores de B
                                        __m256d c_vec = _mm256_loadu_pd(&C(i, j));   // carrega 4 valores de C
                                        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec); // C += A*B
                                        _mm256_storeu_pd(&C(i, j), c_vec);           // armazena resultado
                                    }

                                    // Restante (não múltiplo de 4)
                                    for (; j < j_sub_max; j++) {
                                        C(i, j) += a_ik * B(k, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int N = 1 << 10; // 1024
    const int BLOCK_SIZE = 512;   // blocos grandes (cache L2)
    const int SUB_BLOCK_SIZE = 256; // sub-blocos menores (cache L1)

    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N, N, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    multiply_blocked(A, B, C, BLOCK_SIZE, SUB_BLOCK_SIZE);

    std::cout << "C(0,0) = " << C(0, 0) << '\n';
    return 0;
}
