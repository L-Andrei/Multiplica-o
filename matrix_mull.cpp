#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>
#include <random>
#include <iostream>

using namespace ifnum::linearAlgebra;

template<typename T>
void multiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const int n = A.rows();
    const int m = B.cols();
    const int p = A.cols();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            T sum = 0;
            for (int k = 0; k < p; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

int main(){

    const int N = 1<<10;

    Matrix<double> A(N, N, 0.0);
    Matrix<double> B(N, N, 0.0);
    Matrix<double> C(N,N,0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Preenche A e B com valores aleatórios
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    // Multiplicação simples
    multiply(A, B, C);


    return 0;
}