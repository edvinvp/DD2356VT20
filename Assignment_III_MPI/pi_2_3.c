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
    // MPI Timing

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

    // Handles
    MPI_Request requests[size];
    // Counts for rank 0
    int counts[size];
    // send count to rank 0
    if (rank != 0)
	MPI_Isend(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &requests[rank]);
    else {
	for (int i = 1; i < size; i++)
	    MPI_Irecv(&counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);
    }
    // wait for all requests, remember that no memory is shared.
    // So the sending processes should only wait for their 1 send request
    // while the receiving process has to wait for size - 1 recieve requests
    if (rank != 0)
	MPI_Wait(&requests[rank], MPI_STATUS_IGNORE);
    else 
	MPI_Waitall(size-1, &requests[1], MPI_STATUSES_IGNORE);
    
    // Estimate Pi and display the result
    // Only rank 0
    if (rank == 0) {
	for (int i = 1; i < size; i++)
	    count += counts[i];
	pi = ((double)count / (double)NUM_ITER) * 4.0;
	double end_time = MPI_Wtime();
	printf("The result is %f calculated in  %f seconds\n", pi, end_time - start_time);
    }
    MPI_Finalize();
    
    return 0;
}


