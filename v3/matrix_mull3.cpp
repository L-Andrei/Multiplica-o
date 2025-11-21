#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace ifnum::linearAlgebra;

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


template<typename T>
void multiply_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {

    int blockSize = computeBlockSize(2);
    const int n = A.rows();
    const int m = B.cols();
    const int p = A.cols();

    // Começa a contagem do tempo
    auto start = std::chrono::high_resolution_clock::now();

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int kk = 0; kk < p; kk += blockSize) {

                // Limites reais do bloco (para bordas)
                int i_max = std::min(ii + blockSize, n);
                int j_max = std::min(jj + blockSize, m);
                int k_max = std::min(kk + blockSize, p);

                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        T sum = 0;
                        for (int k = kk; k < k_max; k++) {
                            sum += A(i,k) * B(k, j);
                        }
                        C(i,j) = sum;
                    }
                }
            }
        }
    }
    // Termina a contagem do tempo
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Tempo total: " << elapsed.count() << " s\n";
}

int main() {

    set_real_time_priority();

    const int N = 1 << 10; 

    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N, N, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Preenche A e B com valores aleatórios
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    // Multiplicação por blocos
    multiply_blocked(A, B, C);

    return 0;
}
//Comando utilizado para a compilação:
//g++ -O3 matrix_mull3.cpp -o matrix3