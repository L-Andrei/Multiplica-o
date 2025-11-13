#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>
#include <random>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <chrono>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace ifnum::linearAlgebra;

//------------------------------------------------------------------------------
// Define prioridade máxima e trava memória
//------------------------------------------------------------------------------
void set_real_time_priority() {
    struct sched_param param;
    param.sched_priority = 99; // máxima prioridade possível

    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1)
        perror("Erro ao definir SCHED_FIFO");
    else
        std::cout << "Rodando em tempo real (SCHED_FIFO, prioridade 99)\n";

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1)
        perror("Erro ao travar memória (mlockall)");
    else
        std::cout << "Memória travada (sem paginação)\n";
}

//------------------------------------------------------------------------------
// Multiplicação de matrizes bloqueada com AVX2
//------------------------------------------------------------------------------
template<typename T>
void multiply_blocked(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C,
                      int blockSize, int subBlockSize) {
    const int n = A.rows();
    const int m = B.cols();
    const int p = A.cols();

    auto start = std::chrono::high_resolution_clock::now();

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

                                    int j = jjj;
                                    __m256d a_vec = _mm256_set1_pd(a_ik);

                                    for (; j + 4 <= j_sub_max; j += 4) {
                                        __m256d b_vec = _mm256_loadu_pd(&B(k, j));
                                        __m256d c_vec = _mm256_loadu_pd(&C(i, j));
                                        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                                        _mm256_storeu_pd(&C(i, j), c_vec);
                                    }

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

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Tempo total: " << ns << " ns\n";
}

//------------------------------------------------------------------------------
int main() {
    set_real_time_priority();

    const int N = 1 << 12;          // 1024
    const int BLOCK_SIZE = 512;
    const int SUB_BLOCK_SIZE = 256;

    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N, N, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Warm-up para carregar dados no cache
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }

    multiply_blocked(A, B, C, BLOCK_SIZE, SUB_BLOCK_SIZE);
    return 0;
}
