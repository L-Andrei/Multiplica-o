#include <immintrin.h>
#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace ifnum::linearAlgebra;
using namespace std::chrono;

void set_real_time_priority() {
    struct sched_param param;
    param.sched_priority = 99;

    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1)
        perror("Erro ao definir SCHED_FIFO");
    else
        std::cout << "Rodando em tempo real (SCHED_FIFO, prioridade 99)\n";

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1)
        perror("Erro ao travar memória (mlockall)");
    else
        std::cout << "Memória travada (sem paginação)\n";
}

// Implementação AVX2 manual com intrínsecos
// Cada registrador __m256d contém 4 doubles

template<typename T>
void multiply_blocked_avx2(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C) {

    const size_t blockSize = computeBlockSize(A, 2);
    const size_t subBlockSize = computeBlockSize(A, 1);

    const size_t n = A.rows();
    const size_t m = B.cols();
    const size_t p = A.cols();

    // Criamos Bt para acesso contíguo a colunas de B
    std::vector<T> Bt(m * p);
    for (size_t k = 0; k < p; ++k) {
        for (size_t j = 0; j < m; ++j) {
            Bt[j * p + k] = B(k, j);
        }
    }

    constexpr size_t V = 4; // AVX2 = 4 doubles por registrador

    auto start = high_resolution_clock::now();

    for (size_t ii = 0; ii < n; ii += blockSize) {
        for (size_t jj = 0; jj < m; jj += blockSize) {
            for (size_t kk = 0; kk < p; kk += blockSize) {

                size_t i_max = std::min(ii + blockSize, n);
                size_t j_max = std::min(jj + blockSize, m);
                size_t k_max = std::min(kk + blockSize, p);

                for (size_t iii = ii; iii < i_max; iii += subBlockSize) {
                    for (size_t jjj = jj; jjj < j_max; jjj += subBlockSize) {
                        for (size_t kkk = kk; kkk < k_max; kkk += subBlockSize) {

                            size_t i_sub_max = std::min(iii + subBlockSize, i_max);
                            size_t j_sub_max = std::min(jjj + subBlockSize, j_max);
                            size_t k_sub_max = std::min(kkk + subBlockSize, k_max);

                            for (size_t i = iii; i < i_sub_max; ++i) {

                                for (size_t j = jjj; j < j_sub_max; ++j) {

                                    const T* a_ptr = &A(i, kkk);
                                    const T* b_ptr = &Bt[j * p + kkk];

                                    __m256d acc_vec = _mm256_setzero_pd();
                                    size_t k = kkk;

                                    for (; k + V <= k_sub_max; k += V) {
                                        __m256d va = _mm256_loadu_pd(a_ptr + (k - kkk));
                                        __m256d vb = _mm256_loadu_pd(b_ptr + (k - kkk));
                                        acc_vec = _mm256_fmadd_pd(va, vb, acc_vec);
                                    }

                                    // Redução horizontal do vetor
                                    double tmp[4];
                                    _mm256_storeu_pd(tmp, acc_vec);
                                    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                                    for (; k < k_sub_max; ++k) {
                                        sum += A(i, k) * B(k, j);
                                    }

                                    C(i, j) += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();
    std::cout << "Tempo total (AVX2 intrínsecos): " << ms << " ms\n";
}

int main() {

    set_real_time_priority();

    const size_t N = 1 << 12;

    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N, N, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    multiply_blocked_avx2(A, B, C);

    return 0;
}
//g++ -O3 -mavx2 -mfma teste.cpp -o teste