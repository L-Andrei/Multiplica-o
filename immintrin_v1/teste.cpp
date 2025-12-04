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
        // perror("Erro ao definir SCHED_FIFO"); // Comentado para não poluir se falhar
        ;
    else
        std::cout << "Rodando em tempo real (SCHED_FIFO, prioridade 99)\n";

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1)
        // perror("Erro ao travar memória (mlockall)");
        ;
}

// Multiplicação otimizada para Column-Major (ifnum) usando AVX2 Hand-Written
template<typename T>
void multiply_blocked_avx2(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C) {

    // Nível 2 = Cache L2/L3 (Macro Block)
    const size_t blockSize = computeBlockSize(A, 2); 
    // Nível 1 = Cache L1 (Micro Block/Register Block)
    const size_t subBlockSize = computeBlockSize(A, 1); 

    const size_t n = A.rows();
    const size_t m = B.cols();
    const size_t p = A.cols();

    // Constante para AVX2 (256 bits / 64 bits por double = 4 doubles)
    constexpr size_t V = 4; 

    auto start = high_resolution_clock::now();

    // 1. Loop de Macro-Blocagem (Cache Blocking) - Ordem JJ, KK, II
    for (size_t jj = 0; jj < m; jj += blockSize) {
        for (size_t kk = 0; kk < p; kk += blockSize) {
            for (size_t ii = 0; ii < n; ii += blockSize) {

                size_t i_max = std::min(ii + blockSize, n);
                size_t j_max = std::min(jj + blockSize, m);
                size_t k_max = std::min(kk + blockSize, p);

                // 2. Loop de Sub-Blocagem (Micro-kernel) - Ordem jjj, kkk, iii
                for (size_t jjj = jj; jjj < j_max; jjj += subBlockSize) {
                    for (size_t kkk = kk; kkk < k_max; kkk += subBlockSize) {
                        for (size_t iii = ii; iii < i_max; iii += subBlockSize) {

                            size_t i_sub_max = std::min(iii + subBlockSize, i_max);
                            size_t j_sub_max = std::min(jjj + subBlockSize, j_max);
                            size_t k_sub_max = std::min(kkk + subBlockSize, k_max);

                            // 3. Kernel AVX2 (Column-Major)
                            // Estratégia: Fixa coluna J, carrega bloco vertical de C em registradores,
                            // itera K acumulando (A_col * B_scalar) e salva de volta.
                            
                            for (size_t j = jjj; j < j_sub_max; ++j) {
                                
                                size_t i = iii;

                                // Parte Vetorizada (Passo 4 doubles)
                                for (; i + V <= i_sub_max; i += V) {
                                    
                                    // Ponteiro para a posição C(i, j). Como é Col-Major, 
                                    // os próximos 3 doubles estão em C(i+1, j), C(i+2, j)... (contíguos)
                                    T* c_ptr = &C(i, j);

                                    // Carrega o valor atual de C nos registradores (Accumulator)
                                    __m256d acc_vec = _mm256_loadu_pd(c_ptr);

                                    // Loop K (Produto Interno Otimizado)
                                    for (size_t k = kkk; k < k_sub_max; ++k) {
                                        
                                        // A(i, k) é contíguo verticalmente -> Load Vetorial
                                        __m256d a_vec = _mm256_loadu_pd(&A(i, k));
                                        
                                        // B(k, j) é escalar para toda a coluna 'i' -> Broadcast
                                        // Replica o valor B(k, j) para as 4 posições do vetor
                                        __m256d b_vec = _mm256_set1_pd(B(k, j));

                                        // FMA: acc = (a * b) + acc
                                        acc_vec = _mm256_fmadd_pd(a_vec, b_vec, acc_vec);
                                    }

                                    // Salva o resultado acumulado de volta na memória
                                    _mm256_storeu_pd(c_ptr, acc_vec);
                                }

                                // Parte Escalar (Cleanup para linhas que não formam múltiplo de 4)
                                for (; i < i_sub_max; ++i) {
                                    double sum = C(i, j);
                                    for (size_t k = kkk; k < k_sub_max; ++k) {
                                        sum += A(i, k) * B(k, j);
                                    }
                                    C(i, j) = sum;
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

    const size_t N = 4096;

    // Construtor padrão assumido (Rows, Cols, Val)
    // Se precisar usar Grid2, ajuste aqui conforme sua necessidade,
    // mas evite usar o construtor de cópia quebrado da lib.
    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N, N, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Inicialização
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    multiply_blocked_avx2(A, B, C);
    
    // Verificação visual rápida (evita otimização total do compilador)
    printf("C(0,0): %.5lf\n", C(0,0));

    return 0;
}
// Comando de compilação sugerido:
// g++ -O3 -mavx2 -mfma teste.cpp -o teste
