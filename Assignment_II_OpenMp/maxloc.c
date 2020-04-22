#include "stdlib.h" // rand for instance.
#include <stdio.h>
#include <time.h>
#include <omp.h>

// Question 1
double maxloc_serial() {
    // Array of 1 million doubles
    unsigned int N = 1000000;
    double x[N];
    for(int i=0; i < N;i++){
	// Generate random number between 0 and 1
	x[i] = ((double)(rand()) / RAND_MAX) *
	    ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
    }
    // timing using omp
    double start_time = omp_get_wtime();
    double maxval = 0.0; int maxloc = 0;
    for (int i=0; i < 1000000; i++){
	if (x[i] > maxval) {
	    maxval = x[i];
	    maxloc = i;
	}
    }
    double elapsed_time = omp_get_wtime() - start_time;
    // Avoid compiler optimizing away loops
    printf("%f, %f, %d\n", maxval, x[0], maxloc);
    return  elapsed_time;
}

//Question 2
double maxloc_parallel(void) {
    unsigned int N = 1000000;
    double x[N];
    for(int i=0; i < N;i++) {
	x[i] = ((double)(rand()) / RAND_MAX) *
	    ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
    }
    // timing using omp
    double start_time = omp_get_wtime();
    double maxval = 0.0; int maxloc = 0;
    omp_set_num_threads(32);
    #pragma omp parallel for
    for (int i=0; i < 1000000; i++){
	if (x[i] > maxval)  {
	    maxval = x[i];
	    maxloc = i;
	}
    }
    
    double elapsed_time = omp_get_wtime() - start_time;
    printf("%f, %f, %d\n", maxval, x[0], maxloc);
    return  elapsed_time;
}

// Question 3
double maxloc_parallel_critical(unsigned int num_threads) {
    // Array of 1 million doubles
    unsigned int N = 1000000;
    double x[N];
    for(int i=0; i < N;i++) {
	// Generate random number between 0 and 1
	x[i] = ((double)(rand()) / RAND_MAX) *
	    ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
    }
    // timing using omp
    double start_time = omp_get_wtime();
    double maxval = 0.0; int maxloc = 0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i=0; i < 1000000; i++){
#pragma omp critical
	{
	    if (x[i] > maxval) {
	    maxval = x[i];
	    maxloc = i;
	    }
	}
    }
    
    double elapsed_time = omp_get_wtime() - start_time;
    printf("%f, %f, %d\n", maxval, x[0], maxloc);
    return  elapsed_time;
}


// Question 4
double maxloc_parallel_no_critical(unsigned int num_threads) {
    // Array of 1 million doubles
    unsigned int N = 1000000;
    double x[N];
    for(int i=0; i < N;i++) {
	// Generate random number between 0 and 1
	x[i] = ((double)(rand()) / RAND_MAX) *
	    ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
    }
    // timing using omp
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();
    // avoid using critical by letting each thread find max in their
    // chunk independently
    int maxloc[32];
    double maxval[32];
#pragma omp parallel shared(maxval, maxloc)
    {
    int id = omp_get_thread_num();
    maxval[id] = -1.0e30;
    #pragma omp for
    for (int i=0; i < 1000000; i++){
	if (x[i] > maxval[id]) {
	    maxval[id] = x[i];
	    maxloc[id] = i;
	}
    }
    }
    // serial section
    int s_maxloc = 0; double s_maxval = 0.0;
    for (int i = 0; i < 32; i++) {
	if (maxval[i] > s_maxval) {
	    s_maxval = maxval[i];
	    s_maxloc = maxloc[i];
	}
    }
    double elapsed_time = omp_get_wtime() - start_time;
    printf("%f, %f, %f, %d, %d\n", s_maxval, maxval[0], x[0], s_maxloc, maxloc[0]);
    return  elapsed_time;
}

// Question 5
// One struct per cache line (assume 128 bytes).
typedef struct {
    int loc; // 4 bytes
    double val; // 8 bytes
    char pad[128];
} max_struct;

double maxloc_parallel_with_padding(unsigned int num_threads) {
    unsigned int N = 1000000;
    double x[N];
    for(int i=0; i < N;i++) {
	x[i] = ((double)(rand()) / RAND_MAX) *
	    ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
    }
    omp_set_num_threads(num_threads);
    max_struct maxvals[32];
    double start_time = omp_get_wtime();
#pragma omp parallel shared(maxvals)
    {
    int id = omp_get_thread_num();
    maxvals[id].val = -1.0e30;
    #pragma omp for
    for (int i=0; i < 1000000; i++){
	if (x[i] > maxvals[id].val) {
	    maxvals[id].val = x[i];
	    maxvals[id].loc = i;
	}
    }
    }
    
    int s_maxloc = 0; double s_maxval = -1.0e30;
    for (int i = 0; i < 32; i++) {
	if (maxvals[i].loc > s_maxval) {
	    s_maxval = maxvals[i].val;
	    s_maxloc = maxvals[i].loc;
	}
    }
    
    double elapsed_time = omp_get_wtime() - start_time;
    printf("%f, %f, %f, %d, %d\n", s_maxval, maxvals[0].val, x[0], s_maxloc, maxvals[0].loc);
    return  elapsed_time;
}

int main() {

    // 5 runs per thread count for 3-5, 1-2 is run 5 times each
    double serial_1[5];
    double parallel_2[5];
    double critical_3[10][5];
    double no_critical_4[10][5];
    double no_false_sharing_5[10][5];

    srand(time(0)); // seed

    unsigned int threads[10] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32};
    for (int i = 0; i < 5; i++) {
	serial_1[i] = maxloc_serial();
	parallel_2[i] = maxloc_parallel();
    }

    for (int i = 0; i < 10; i++) {
	for (int j = 0; j < 5; j++) {
	    critical_3[i][j] = maxloc_parallel_critical(threads[i]);
	    no_critical_4[i][j] = maxloc_parallel_no_critical(threads[i]);
	    no_false_sharing_5[i][j] = maxloc_parallel_with_padding(threads[i]);
	}
    }

    printf("\n\nQUESTION 1: SERIAL\n");
    for (int i = 0; i < 5; i++)
	i != 4 ? printf("%f, ", serial_1[i]) : printf("%f\n\n", serial_1[i]);
    
    printf("QUESTION 2: PARALLEL 32 THREADS\n");
    for (int i = 0; i < 5; i++)
	i != 4 ? printf("%f, ", parallel_2[i]) : printf("%f\n\n", parallel_2[i]);

    printf("QUESTION 3: CRITICAL\n");
    for (int i = 0; i < 10; i++) {
	printf("Q3_%d = [", threads[i]);
	for (int j = 0; j < 5; j++)
	    j != 4 ? printf("%f, ", critical_3[i][j]) : printf("%f]\n", critical_3[i][j]);
    }

    printf("QUESTION 4: NO CRITICAL\n");
    for (int i = 0; i < 10; i++) {
	printf("Q4_%d = [", threads[i]);
	for (int j = 0; j < 5; j++)
	    j != 4 ? printf("%f, ", no_critical_4[i][j]) : printf("%f]\n", no_critical_4[i][j]);
    }
    printf("QUESTION 5: NO FALSE SHARING\n");
    for (int i = 0; i < 10; i++) {
	printf("Q5_%d = [", threads[i]);
	for (int j = 0; j < 5; j++)
	    j != 4 ? printf("%f, ", no_false_sharing_5[i][j]) : printf("%f]\n", no_false_sharing_5[i][j]);
    }
    return 0;
}
