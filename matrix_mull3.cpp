#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>
#include <random>
#include <iostream>
#include <algorithm>

using namespace ifnum::linearAlgebra;

// Multiplicação de matrizes com blocagem
template<typename T>
void multiply_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int blockSize) {
    const int n = A.rows();
    const int m = B.cols();
    const int p = A.cols();

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int kk = 0; kk < p; kk += blockSize) {

                // Limites reais do bloco (para bordas)
                int i_max = std::min(ii + blockSize, n);
                int j_max = std::min(jj + blockSize, m);
                int k_max = std::min(kk + blockSize, p);

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        T a_ik = A(i, k);
                        for (int j = jj; j < j_max; j++) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int N = 1 << 10; 
    const int BLOCK_SIZE = 256; 

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
    multiply_blocked(A, B, C, BLOCK_SIZE);

    std::cout << "C(0,0) = " << C(0, 0) << '\n';
    return 0;
}
