#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    int count = 0;
    double x, y, z, pi;
    int rank, size, provided;
    // Start MPI
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    double start_time = MPI_Wtime();
    // get rank of current process and number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(SEED * rank); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    // Calculate PI following a Monte Carlo metcd hod
    for (int iter = rank; iter < NUM_ITER; iter += size) {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0) {
            count++;
        }
    }

    // Binary reduction
    int height = log2(size);
    int active[size];
    // set active to 1's
    for (int i = 0; i < size; i++)
	active[i] = 1;
    for (int k = 0; k < height; k++) {
	for (int i = 0; i < size; i++) {
	    if (active[i] && rank == i) {
		// 2^k by right shift
		int k_2 = 1 << k;
		// check if bit k of i is set (= 1)
		if ((i>>k) & 1) {
		    // Then send to process i-2^k
		    int dest = i - k_2;
		    MPI_Send(&count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		    // after we have sent set process k to inactive
		    active[k] = 0;
		} else if (i + k_2 < size) {
		    int tmp_count = 0;
		    int source = i + k_2;
		    MPI_Recv(&tmp_count, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    count += tmp_count;
		}
	    }
	}
    }
    
    // Estimate Pi and display the result
    // Only rank 0
    if (rank == 0) {
	pi = ((double)count / (double)NUM_ITER) * 4.0;
	double end_time = MPI_Wtime();
	printf("The result is %f calculated in  %f seconds\n", pi, end_time - start_time);
    }
    MPI_Finalize();
    
    return 0;
}


