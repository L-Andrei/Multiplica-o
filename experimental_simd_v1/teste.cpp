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
#include <experimental/simd>

using namespace ifnum::linearAlgebra;
using namespace std::chrono;
namespace simd = std::experimental;

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

// Multiplicação de matrizes com blocagem hierárquica e vetorizações usando
// std::experimental::simd (especificado para largura AVX2: 4 doubles por reg)
template<typename T>
void multiply_blocked_simd(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C) {

    const size_t blockSize = computeBlockSize(A, 2);
    const size_t subBlockSize = computeBlockSize(A, 1);

    const size_t n = A.rows();
    const size_t m = B.cols();
    const size_t p = A.cols();

    // Preparar uma versão transposta de B (Bt) para acesso contíguo a colunas de B
    // Bt tem dimensões m x p, cada linha de Bt = uma coluna de B
    std::vector<T> Bt(m * p);
    for (size_t k = 0; k < p; ++k) {
        for (size_t j = 0; j < m; ++j) {
            Bt[j * p + k] = B(k, j);
        }
    }

    // Define o tipo SIMD: 4 doubles (256 bits) — adequado para AVX/AVX2
    using abi = simd::simd_abi::fixed_size<4>;
    using simd_t = simd::simd<T, abi>;
    constexpr size_t V = simd_t::size(); // tipicamente 4 para double

    auto start = high_resolution_clock::now();

    for (size_t ii = 0; ii < n; ii += blockSize) {
        for (size_t jj = 0; jj < m; jj += blockSize) {
            for (size_t kk = 0; kk < p; kk += blockSize) {

                size_t i_max = std::min(ii + blockSize, n);
                size_t j_max = std::min(jj + blockSize, m);
                size_t k_max = std::min(kk + blockSize, p);

                // Sub-blocos
                for (size_t iii = ii; iii < i_max; iii += subBlockSize) {
                    for (size_t jjj = jj; jjj < j_max; jjj += subBlockSize) {
                        for (size_t kkk = kk; kkk < k_max; kkk += subBlockSize) {

                            size_t i_sub_max = std::min(iii + subBlockSize, i_max);
                            size_t j_sub_max = std::min(jjj + subBlockSize, j_max);
                            size_t k_sub_max = std::min(kkk + subBlockSize, k_max);

                            // Para cada elemento do sub-bloco (i,j) acumulamos sobre k
                            for (size_t i = iii; i < i_sub_max; ++i) {
                                for (size_t j = jjj; j < j_sub_max; ++j) {
                                    // ponteiro para a linha i de A começando em kkk
                                    const T* a_ptr = &A(i, kkk);
                                    // ponteiro para a "linha" j de Bt (que é a coluna j de B)
                                    const T* b_ptr = &Bt[j * p + kkk];

                                    // soma vetor e escalar para o resto
                                    simd_t acc_vec(0);
                                    size_t k = kkk;

                                    // loop vetorizado: processa V elementos por iteração
                                    for (; k + V <= k_sub_max; k += V) {
                                        // carrega V elementos contíguos de A e Bt (coluna de B)
                                        simd_t va(a_ptr + (k - kkk), simd::element_aligned);
                                        simd_t vb(b_ptr + (k - kkk), simd::element_aligned);
                                        acc_vec += va * vb;
                                    }

                                    // reduzir o vetor acumulador para um escalar
                                    T sum = 0;
                                    for (size_t t = 0; t < V; ++t) sum += acc_vec[t];

                                    // resto não múltiplo de V
                                    for (; k < k_sub_max; ++k) {
                                        sum += A(i, k) * B(k, j);
                                    }

                                    C(i, j) += sum; // nota: acumulamos ao invés de sobrescrever
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
    std::cout << "Tempo total (SIMD): " << ms << " ms\n";
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

    // Executa a multiplicação vetorizada
    multiply_blocked_simd(A, B, C);

    return 0;
}
//g++ -O3 -mavx2 teste.cpp -o teste