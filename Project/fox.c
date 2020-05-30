#include <mpi.h>
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
			    int j = jj;
			    C[i*n + j + 0] += A[i*n + k] * B[k*n + j + 0];
			    C[i*n + j + 1] += A[i*n + k] * B[k*n + j + 1];
			    C[i*n + j + 2] += A[i*n + k] * B[k*n + j + 2];
			    C[i*n + j + 3] += A[i*n + k] * B[k*n + j + 3];
			    C[i*n + j + 4] += A[i*n + k] * B[k*n + j + 4];
			    C[i*n + j + 5] += A[i*n + k] * B[k*n + j + 5];
			    C[i*n + j + 6] += A[i*n + k] * B[k*n + j + 6];
			    C[i*n + j + 7] += A[i*n + k] * B[k*n + j + 7];
			    C[i*n + j + 8] += A[i*n + k] * B[k*n + j + 8];
			    C[i*n + j + 9] += A[i*n + k] * B[k*n + j + 9];
			    C[i*n + j + 10] += A[i*n + k] * B[k*n + j + 10];
			    C[i*n + j + 11] += A[i*n + k] * B[k*n + j + 11];
			    C[i*n + j + 12] += A[i*n + k] * B[k*n + j + 12];
			    C[i*n + j + 13] += A[i*n + k] * B[k*n + j + 13];
			    C[i*n + j + 14] += A[i*n + k] * B[k*n + j + 14];
			    C[i*n + j + 15] += A[i*n + k] * B[k*n + j + 15];
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

int main(int argc, char* argv[])
{
    // Assumptions:
    // 1. Matrices are square
    // 2. Number of processes must be a square
    // 3. The size depends on the number of processes

    /*
     * The Fox algorithm A*B = C implemented using cartesian communicators.
     * For the A broadcast a cartesian row communicator is used and for the
     * 'Roll' step the full cartesian grid.
     */
    int rank, num_processes, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    omp_set_num_threads(4);
    /*
     * We will perform benchmarks using even powers of 2 number of processes.
     * The block size is then given by dividing the matrix size sqrt(num_processes)
     * If using the mat_mul_strip BLOCK_SIZE must be a multiple of 16.
     */
    int MATRIX_SIZE = pow(2, 10); // 4096, 2^6 = 64
    int BLOCK_SIZE = MATRIX_SIZE / (int) sqrt(num_processes); // EX: with 4 processes and 2^12 matrix size => 2^10 block size
    float *A_full = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    float *B_full = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    float *C_full = malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    test_matrices_2(A_full, B_full, C_full, MATRIX_SIZE);

    // Matrix block setup. Know that num_processes is square.
    float *A_block_orig = malloc(sizeof(float) * BLOCK_SIZE * BLOCK_SIZE);
    float *A_block = malloc(sizeof(float) * BLOCK_SIZE * BLOCK_SIZE);
    float *B_block = malloc(sizeof(float) * BLOCK_SIZE * BLOCK_SIZE);
    float *C_block = malloc(sizeof(float) * BLOCK_SIZE * BLOCK_SIZE);   
    
    // Create cartesian toplogy
    MPI_Comm cart_communicator, row_communicator;
    int sqrt_num_processes = sqrt(num_processes);
    int dim_size = sqrt_num_processes;
    int dims[2] = {dim_size, dim_size};
    // Specify row sub communicator dimensions, MPI uses (col,row) indexing
    // By fixing the column dimension sub communicators are created for each row
    int sub_dims[2] = {1,0}; 
    // The row does not have to be periodic as we use a sub commincator and
    // MPI_Bcast for the rows.
    int periods[2] = {1,1};
    // 2 = num dims, 1 = MPI can reorder ranks
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_communicator);
    MPI_Cart_sub(cart_communicator, sub_dims, &row_communicator);

    int cart_size, cart_rank, row_size, row_rank;
    int cart_coords[2];
    MPI_Comm_size(cart_communicator, &cart_size);
    MPI_Comm_rank(cart_communicator, &cart_rank);
    MPI_Comm_size(row_communicator, &row_size);
    MPI_Comm_rank(row_communicator, &row_rank);
    MPI_Cart_coords(cart_communicator, cart_rank, 2, cart_coords);    

    // Copy large matrix into blocks. MPI uses (col, row).
    // The start of the processor block is the cart coordinates offset by the block size
    int i_block = cart_coords[1] * BLOCK_SIZE;
    int j_block = cart_coords[0] * BLOCK_SIZE;
    for (int i = 0; i < BLOCK_SIZE; i++) {
	for (int j = 0; j < BLOCK_SIZE; j++) {
	    // The number of cols = number of processes
	    A_block_orig[i*BLOCK_SIZE + j] = A_full[(MATRIX_SIZE)*(i_block + i) + (j_block + j)];
	    A_block[i*BLOCK_SIZE + j] = A_full[(MATRIX_SIZE)*(i_block + i) + (j_block + j)];
	    B_block[i*BLOCK_SIZE + j] = B_full[(MATRIX_SIZE)*(i_block + i) + (j_block + j)];
	    C_block[i*BLOCK_SIZE + j] = 0.0f;
	}
    }

    // Find column neighbors for the B matrix column roll
    int roll_source, roll_dest, roll_recv;
    // 1 = coordinate dimension of shift (1 = row)
    // -1 = displacement (> 0 up, < 0 down)
    // roll_source is the rank of the corrent process, roll_dest is the one we shift too
    // Since we roll 'up' we have -1 as the destination shift.
    MPI_Cart_shift(cart_communicator, 1, -1, &roll_source, &roll_dest);
    // Recieve B from
    MPI_Cart_shift(cart_communicator, 1, 1, &roll_source, &roll_recv);
    MPI_Status roll_status;
    
    // Timing: TODO: Ask if it is reasonable to start the timing here or if I should create the
    // matrix on the root and the broadcast each block to the corresponding proocess.
    // Why I start here now: According to the article we often want to keep the block structure
    // to perform additional matirx operations. I think that these operations are most likely
    // multiplications.
    double start_time = MPI_Wtime();
    
    // The fox algorithm
    for (int iteration = 0; iteration < sqrt_num_processes; ++iteration) {
	// Step 1. Broadcast Diagonal offset by iteration
	int root_rank;
	int root_coords[2] = {(cart_coords[1] + iteration) % sqrt_num_processes, cart_coords[1]};
	MPI_Cart_rank(row_communicator, root_coords, &root_rank);
	
	// Check if current proccess is root
	if (cart_coords[0] == root_coords[0] && cart_coords[1] == root_coords[1]) {
	    MPI_Bcast(A_block_orig, BLOCK_SIZE * BLOCK_SIZE, MPI_FLOAT, root_rank, row_communicator);
	} else {
	    MPI_Bcast(A_block, BLOCK_SIZE * BLOCK_SIZE, MPI_FLOAT, root_rank, row_communicator);
	}
	
	// Step 2. Multiply Recieved A with Current B
	// if root use original A block
	if (cart_coords[0] == root_coords[0] && cart_coords[1] == root_coords[1]) {
	    mat_mul_strip(A_block_orig, B_block, C_block, BLOCK_SIZE);
	} else {
	    mat_mul_strip(A_block, B_block, C_block, BLOCK_SIZE);
	}

	// Step 3. "Roll" B "up" one step.
	// We can stop here if last iteration.
	if (iteration  == sqrt_num_processes - 1) {
	    break;
	}
	MPI_Sendrecv_replace(B_block, BLOCK_SIZE*BLOCK_SIZE, MPI_FLOAT, roll_dest, 0, roll_recv, 0, cart_communicator, &roll_status);
    }

    double elapsed_time = MPI_Wtime() - start_time;
    if (rank == 0) {
	printf("Fox algorithm time: %f \n", elapsed_time);
    }
    printf("(%d, %d): %f, %f, %f, %f, %f\n", cart_coords[0], cart_coords[1], C_block[0], C_block[1], C_block[2], C_block[3], C_block[4]);

    // reference
    double mat_mul_time = MPI_Wtime();
    mat_mul_strip(A_full, B_full, C_full, MATRIX_SIZE);
    double mat_mul_elapsed = MPI_Wtime() - mat_mul_time;
    if (rank == 0) {
	printf("mat_mul time: %f \n", mat_mul_elapsed);
    }

    // reset C_Full
    for (int i = 0; i < MATRIX_SIZE; i++) {
	for (int j = 0; j < MATRIX_SIZE; j++) {
	    C_full[i*MATRIX_SIZE + j] = 0.0f;
	}
    }
    double mat_mul_slow_time = MPI_Wtime();
    // mat_mul_naive(A_full, B_full, C_full, MATRIX_SIZE);
    double mat_mul_slow_elapsed = MPI_Wtime() - mat_mul_slow_time;
    if (rank == 0) {
//	printf("mat_mul time: %f \n", mat_mul_slow_elapsed);
    }
    
    
    free(A_block_orig);
    free(A_block); free(B_block); free(C_block);
    free(A_full); free(B_full); free(C_full);
    
    MPI_Finalize();
    
    return 0;
}
