/* @Edvin von Platen
 * O(n^3) matrix multiplication algorithm with different
 * optimization levels.
 */

#include <stdio.h>
#include <math.h> // sqrt, compile with -lm
#include <string.h> // memcpy
#include <omp.h>
#include <stdlib.h> // free

/* Multiply the square matrices A*B = C */
void mat_mul_reorder_unroll_omp(float *A, float *B, float *C, int n) {
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; ++i) {
	for (int k = 0; k < n; ++k) {
	    for (int j = 0; j < n; j += 4) {
		C[i*n + j + 0] += A[i*n + k] * B[k*n + j + 0];
		C[i*n + j + 1] += A[i*n + k] * B[k*n + j + 1];
		C[i*n + j + 2] += A[i*n + k] * B[k*n + j + 2];
		C[i*n + j + 3] += A[i*n + k] * B[k*n + j + 3];
	    }
	}
    }
}

/* Multiply the square matrices A*B = C */
void mat_mul_reorder_unroll(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; ++i) {
	for (int k = 0; k < n; ++k) {
	    for (int j = 0; j < n; j += 4) {
		C[i*n + j + 0] += A[i*n + k] * B[k*n + j + 0];
		C[i*n + j + 1] += A[i*n + k] * B[k*n + j + 1];
		C[i*n + j + 2] += A[i*n + k] * B[k*n + j + 2];
		C[i*n + j + 3] += A[i*n + k] * B[k*n + j + 3];
	    }
	}
    }
}

/* Multiply the square matrices A*B = C */
void mat_mul_reorder(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; ++i) {
	for (int k = 0; k < n; ++k) {
	    for (int j = 0; j < n; ++j) {
		C[i*n + j] += A[i*n + k] * B[k*n + j];
	    }
	}
    }
}

void mat_mul_strip(float *A, float *B, float *C, int n) {
    // Cache line 64 bytes: Assume 4 bytes per float, using 64/sizeof(float) could break the unroll.    
    const int STRIP = 16;
    // Requires the matrix row size to be a multiple of 16.
    // If not use a non-blocked and non-unrolled method or chan
#pragma omp parallel for schedule(guided)
    for (int ii = 0; ii < n; ii+=STRIP) {
	for (int kk = 0; kk < n; kk+=STRIP) {
	    for (int jj = 0; jj < n; jj += STRIP) {
		for (int i = ii; i < ii + STRIP; ++i) {
		    for (int k = kk; k < kk + STRIP; ++k) {
			    C[i*n + jj + 0] += A[i*n + k] * B[k*n + jj + 0];
			    C[i*n + jj + 1] += A[i*n + k] * B[k*n + jj + 1];
			    C[i*n + jj + 2] += A[i*n + k] * B[k*n + jj + 2];
			    C[i*n + jj + 3] += A[i*n + k] * B[k*n + jj + 3];
			    C[i*n + jj + 4] += A[i*n + k] * B[k*n + jj + 4];
			    C[i*n + jj + 5] += A[i*n + k] * B[k*n + jj + 5];
			    C[i*n + jj + 6] += A[i*n + k] * B[k*n + jj + 6];
			    C[i*n + jj + 7] += A[i*n + k] * B[k*n + jj + 7];
			    C[i*n + jj + 8] += A[i*n + k] * B[k*n + jj + 8];
			    C[i*n + jj + 9] += A[i*n + k] * B[k*n + jj + 9];
			    C[i*n + jj + 10] += A[i*n + k] * B[k*n + jj + 10];
			    C[i*n + jj + 11] += A[i*n + k] * B[k*n + jj + 11];
			    C[i*n + jj + 12] += A[i*n + k] * B[k*n + jj + 12];
			    C[i*n + jj + 13] += A[i*n + k] * B[k*n + jj + 13];
			    C[i*n + jj + 14] += A[i*n + k] * B[k*n + jj + 14];
			    C[i*n + jj + 15] += A[i*n + k] * B[k*n + jj + 15];
		    }
		}
	    }
	}
    }    
}

void mat_mul_naive(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    for (int k = 0; k < n; k++) {
		C[i*n + j] += A[i*n +k] * B[k*n + j];
	    }
	}
    }
}

/* First test: A n x n  matrix of 1's and B n x n matrix of 2's
 * The product AB = C then ýields a n x n matrix C where all
 * entries are equal to 2*n */
void test_matrices_1(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    A[i*n + j] = 1.0f;
	    B[i*n + j] = 2.0f;
	    C[i*n + j] = 0.0f;
	}
    }
}

/* A: diagonal with the row number on the diagonal
 * B[i,j] = i * j
 * C[i,j] = i^2 * j
 */
void test_matrices_2(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    A[i*n + j] = (i == j) ? (float) i : 0.0f;
	    B[i*n + j] = (float) i * j;
	    C[i*n + j] = 0.0f;
	}
    }
}

void set_zero(float *mat, int n)
{
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    mat[i*n + j] = 0.0f;
	}
    }
}

void print_diag(float *mat, int n) {
    printf("DIAG: [");
    for (int i = 0; i < n - 1; i++) {
	printf("%.2f, ", mat[i*n + i]);
    }
    printf("%.2f]\n", mat[n*n - 1]);
}

// Checks if A = B within eps precision.
int same_matrices(float *A, float *B, int n) {
    float eps = 0.000001f;
    for (int i = 0; i < n; ++i) {
	for (int j = 0; j < n; ++j) {
	    if (fabs((float)A[i*n + j] - (float) B[i*n + j]) > eps) {
		return 0;
	    }
	}
    }
    return 1;	
}

void test_mat_mult(float *A, float *B, float *C, float *C_ref, int n)
{
    int check = 0;   
    mat_mul_naive(A,B,C_ref, n);
    
    mat_mul_reorder(A, B, C, n);
    check = same_matrices(C_ref, C, n);
    if (!check) {
	printf("mat_mul_reorder wrong\n");
    }
    set_zero(C, n);
    
    mat_mul_reorder_unroll(A, B, C, n);
    check = same_matrices(C_ref, C, n);
    if (!check) {
	printf("mat_mul_reorder_unroll wrong\n");
    }
    set_zero(C, n);
    
    mat_mul_reorder_unroll_omp(A, B, C, n);
    check = same_matrices(C_ref, C, n);
    if (!check) {
	printf("mat_mul_reorder_omp wrong\n");
    }
    set_zero(C, n);
    
    mat_mul_strip(A, B, C, n);
    check = same_matrices(C_ref, C, n);
    if (!check) {
	printf("mat_mul_strip wrong\n");
    }
}


// Takes the matrix multiplication function and its parameters as arguments
// Returns the average runtime of 'runs' runs.
double benchmark_mat_mult(int runs, void (*mat_mult)(float *, float *, float *, int), float * A, float *B, float *C, int n)
{
    double avg = 0.0;
    for (int i = 0; i < runs; ++i) {
	double start_time = omp_get_wtime();
        (*mat_mult)(A, B, C, n);
        double elapsed_time =  omp_get_wtime() - start_time;
	avg += elapsed_time;
	set_zero(C, n);	
    }
    return avg / (double) runs;	
}

int main(int argc, char* argv[])
{
    int MATRIX_SIZE = pow(2, 10);
    float *A = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    float *B = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    float *C = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    test_matrices_1(A, B, C, MATRIX_SIZE);

    float *C_ref = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    set_zero(C_ref, MATRIX_SIZE);

    omp_set_num_threads(4);

    // Uncomment to test implemented matrix algorithms with
    // the naive implementation as reference
    printf("Running tests\n");
    test_mat_mult(A,B,C,C_ref,MATRIX_SIZE);
    set_zero(C, MATRIX_SIZE);

    // Benchmark each matrix multiplication implementation
    int runs = 10;
    printf("Benchmark\n");
    printf("Avg of %d runs\n", runs);
    for (int i = 13; i < 13; ++i) {
	int n = pow(2, i);
	float *A_bench = malloc(sizeof(float) * n * n);
	float *B_bench = malloc(sizeof(float) * n * n);
	float *C_bench = malloc(sizeof(float) * n * n);
	test_matrices_1(A_bench, B_bench, C_bench, n);
	
	double avg_naive = benchmark_mat_mult(runs, mat_mul_naive, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Naive avg time = %.6f s\n",n, avg_naive);
	
	double avg_reorder = benchmark_mat_mult(runs, mat_mul_reorder, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Reorder avg time = %.6f s\n",n, avg_reorder);

	double avg_reorder_unroll = benchmark_mat_mult(runs, mat_mul_reorder_unroll, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Reorder_Unroll avg time = %.6f s\n",n, avg_reorder_unroll);

	double avg_reorder_unroll_omp = benchmark_mat_mult(runs, mat_mul_reorder_unroll_omp, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Reorder_Unroll_Omp avg time = %.6f s\n",n, avg_reorder_unroll_omp);

	double avg_strip = benchmark_mat_mult(runs, mat_mul_strip, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Strip avg time = %.6f s\n",n, avg_strip);
	
	free(A_bench); free(B_bench); free(C_bench);
	printf("\n");
    }

    
    // Test with 8 threads also
    printf("8 Threads Test\n");
    omp_set_num_threads(8);
    for (int i = 5; i < 13; ++i) {
	int n = pow(2, i);
	float *A_bench = malloc(sizeof(float) * n * n);
	float *B_bench = malloc(sizeof(float) * n * n);
	float *C_bench = malloc(sizeof(float) * n * n);
	test_matrices_1(A_bench, B_bench, C_bench, n);
        
	double avg_reorder_unroll_omp = benchmark_mat_mult(runs, mat_mul_reorder_unroll_omp, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Reorder_Unroll_Omp avg time = %.6f s\n",n, avg_reorder_unroll_omp);

	double avg_strip = benchmark_mat_mult(runs, mat_mul_strip, A_bench, B_bench, C_bench, n);
	set_zero(C_bench, n);
	printf("n = %d, Strip avg time = %.6f s\n",n, avg_strip);
	
	free(A_bench); free(B_bench); free(C_bench);
	printf("\n");
    }
    
    free(A); free(B); free(C), free(C_ref);
}
