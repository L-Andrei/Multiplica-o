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

// Namespaces
using namespace ifnum::linearAlgebra;
namespace simd = std::experimental;

// --- Configurações de Sistema (Prioridade Real-Time) ---
void set_real_time_priority() {
    struct sched_param param;
    param.sched_priority = 99;
    // Ignora erros silenciosamente se não tiver sudo
    sched_setscheduler(0, SCHED_FIFO, &param);
    mlockall(MCL_CURRENT | MCL_FUTURE);
}

// --- Multiplicação Otimizada para Column-Major (Sua Lib) ---
template<typename T>
void multiply_simd_col_major(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C) {
    
    const size_t n = A.rows();
    const size_t m = B.cols();
    const size_t p = A.cols(); 

    // Nível 1: Macro-Bloco (Para Cache L2/L3) - ex: 256 ou 512
    const size_t blockSize = computeBlockSize(A, 2); 
    
    // Nível 2: Sub-Bloco (Para Cache L1) - ex: 32 ou 64
    // Assumindo que computeBlocksize(1) retorne o tamanho ideal para L1
    // Se não tiver essa função, use fixo: const size_t subBlockSize = 64;
    const size_t subBlockSize = computeBlockSize(A, 2); 

    using abi = simd::simd_abi::fixed_size<4>;
    using simd_t = simd::simd<T, abi>;
    constexpr size_t V = simd_t::size();

    auto start = std::chrono::high_resolution_clock::now();

    // --- LOOP NÍVEL 1: Macro-Blocagem (Cache Blocking) ---
    for (size_t jj = 0; jj < m; jj += blockSize) {
        for (size_t kk = 0; kk < p; kk += blockSize) {
            for (size_t ii = 0; ii < n; ii += blockSize) {

                // Limites do Macro-Bloco
                size_t i_max = std::min(ii + blockSize, n);
                size_t j_max = std::min(jj + blockSize, m);
                size_t k_max = std::min(kk + blockSize, p);

                // --- LOOP NÍVEL 2: Sub-Blocagem (Micro-Kernel Blocking) ---
                for (size_t jjj = jj; jjj < j_max; jjj += subBlockSize) {
                    for (size_t kkk = kk; kkk < k_max; kkk += subBlockSize) {
                        for (size_t iii = ii; iii < i_max; iii += subBlockSize) {

                            // Limites do Sub-Bloco
                            size_t i_sub_max = std::min(iii + subBlockSize, i_max);
                            size_t j_sub_max = std::min(jjj + subBlockSize, j_max);
                            size_t k_sub_max = std::min(kkk + subBlockSize, k_max);

                            // --- KERNEL DE CÁLCULO (Registradores / AVX) ---
                            // Mantemos a estratégia Column-Major: J externo, I vetorizado
                            for (size_t j = jjj; j < j_sub_max; ++j) {
                                
                                size_t i = iii;

                                // 1. Parte Vetorizada (AVX)
                                for (; i + V <= i_sub_max; i += V) {
                                    
                                    T* c_ptr = &C(i, j);
                                    simd_t acc_vec(c_ptr, simd::element_aligned);

                                    // Loop K interno (reduz load/store de C)
                                    for (size_t k = kkk; k < k_sub_max; ++k) {
                                        const T* a_ptr = &A(i, k);
                                        simd_t vec_a(a_ptr, simd::element_aligned);
                                        
                                        T val_b = B(k, j); 
                                        acc_vec += vec_a * val_b;
                                    }
                                    acc_vec.copy_to(c_ptr, simd::element_aligned);
                                }

                                // 2. Parte Escalar (Resto do sub-bloco)
                                for (; i < i_sub_max; ++i) {
                                    T sum = C(i, j); 
                                    for (size_t k = kkk; k < k_sub_max; ++k) {
                                        sum += A(i, k) * B(k, j);
                                    }
                                    C(i, j) = sum;
                                }
                            }
                            // Fim do Kernel
                        }
                    }
                }
                // Fim do Loop Nível 2
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tempo total (SIMD Blocked Col-Major): " << ms << " ms\n";
}

// Verifica integridade matemática
bool verify(Matrix<double>& A, Matrix<double>& B, Matrix<double>& C) {
    size_t N = A.rows();
    std::cout << "Verificando amostra...\n";
    // Verifica 3 posições aleatórias
    for(int k=0; k<3; k++) {
        size_t r = rand() % N;
        size_t c = rand() % N;
        double acc = 0;
        for(size_t i=0; i<N; i++) acc += A(r, i) * B(i, c);
        
        if (std::abs(acc - C(r, c)) > 1e-9) {
            std::cout << "Erro em (" << r << "," << c << "): " 
                      << "Calc=" << C(r, c) << " Real=" << acc << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    set_real_time_priority();

    const size_t N = 4096; // Matriz 1024x1024

    std::cout << "Alocando matrizes...\n";

    // CORREÇÃO AQUI: Instancia direto, sem passar pelo Grid2
    // Assumindo construtor Matrix(rows, cols, initial_value)
    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N, N, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Preenchimento
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    std::cout << "Iniciando multiplicacao...\n";
    multiply_simd_col_major(A, B, C);

    if(verify(A, B, C)) {
        std::cout << "Resultado: SUCESSO\n";
    } else {
        std::cout << "Resultado: FALHA\n";
    }
    
    // Evita otimização excessiva imprimindo um valor real
    std::cout << "Check: " << C(0,0) << std::endl;

    return 0;
}
