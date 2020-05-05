#include "stdio.h" // printf
#include "stdlib.h" // malloc and rand for instance. Rand not thread safe!
#include "time.h"   // time(0) to get random seed
#include "math.h"  // sine and cosine
#include "omp.h"   // openmp library like timing
#include "string.h" // memset

// Psuedo code
/*
1 Get input data; 
2 for each timestep { 
3      if (timestep output) Print positions and velocities of particles; 
4      for each particle q 
5         Compute total force on q; 
6      for each particle q 
7         Compute position and velocity of q; 
8 } 
9 Print positions and velocities of particles;
 */

// For lines 4-5
/*
for each particle q { 
    for each particle k != q { 
            x_diff = pos[q][X] − pos[k][X]; 
            y_diff = pos[q][Y] − pos[k][Y]; 
            dist = sqrt(x_diff∗x_diff + y_diff∗y_diff); 
            dist_cubed = dist∗dist∗dist; 
            forces[q][X] −= G∗masses[q]∗masses[k]/dist_cubed ∗ x_diff; 
            forces[q][Y] −= G∗masses[q]∗masses[k]/dist_cubed ∗ y_diff; 
    } 
}
*/

// Constants / Input
// Instead of calculating steps by T/DT I use a variable as we are
// asked to run for 100 "cycles"
#define N 1000
#define T 10
#define DT 0.05
#define STEPS 100
#define G 6.673E-10
#define DIM 2
#define X 0
#define Y 1

// (X,Y) values
typedef double vect_t[DIM];

double simple_algorithm(vect_t *pos, vect_t *old_pos, vect_t *vel, double *mass) {
    double start_time = omp_get_wtime();
    vect_t *forces = malloc(N*sizeof(vect_t));
    for (unsigned int i = 0; i < STEPS; i++) {
	// Forces array set to 0
	forces = memset(forces, 0, N*sizeof(vect_t));
	// Compute forces
#pragma omp parallel for schedule(guided)
	for (unsigned int q = 0; q < N; q++) {
	    // Divide the computation in two loops, slicing forces on q
	    // to avoid an if clause inside a single loop
	    double x_diff = 0.0;
	    double y_diff = 0.0;
	    double dist = 0.0;
	    double dist_cubed = 0.0;
	    // k < q
	    for (unsigned int k = 0; k < q; k++) {
	        x_diff = pos[q][X] - pos[k][X]; 
		y_diff = pos[q][Y] - pos[k][Y]; 
	        dist = sqrt(x_diff*x_diff + y_diff*y_diff); 
		dist_cubed = dist*dist*dist; 
		forces[q][X] -= G*mass[q]*mass[k]/dist_cubed * x_diff; 
		forces[q][Y] -= G*mass[q]*mass[k]/dist_cubed * y_diff; 
	    }
	    // k > q
	    for (unsigned int k = q + 1; k < N; k++) {
		x_diff = pos[q][X] - pos[k][X]; 
		y_diff = pos[q][Y] - pos[k][Y]; 
		dist = sqrt(x_diff*x_diff + y_diff*y_diff); 
		dist_cubed = dist*dist*dist; 
		forces[q][X] -= G*mass[q]*mass[k]/dist_cubed * x_diff; 
		forces[q][Y] -= G*mass[q]*mass[k]/dist_cubed * y_diff; 
	    }
	}

	// Update velocities
	#pragma omp parallel for schedule(guided)
	for (unsigned int q = 0; q < N; q++) {
	    pos[q][X] += DT*vel[q][X];
	    pos[q][Y] += DT*vel[q][Y];
	    vel[q][X] += DT/mass[q]*forces[q][X];
	    vel[q][Y] += DT/mass[q]*forces[q][Y];	   	    
	}
    }
    double elapsed_time = omp_get_wtime() - start_time;
    #if PRINT
    for (unsigned int q = 0; q < N; q++)
	q == N - 1 ? printf("(%f, %f)\n", pos[q][X], pos[q][Y]) : printf("(%f, %f)", pos[q][X], pos[q][Y]);
    #endif
    free(forces);
    return elapsed_time;
}
double reduced_algorithm(vect_t *pos, vect_t *old_pos, vect_t *vel, double *mass) {
    double start_time = omp_get_wtime();
    vect_t *forces = malloc(N*sizeof(vect_t));
    int t = 1;
    for (unsigned int i = 0; i < STEPS; i++) {
	// Forces array set to 0
	forces = memset(forces, 0, N*sizeof(vect_t));
	// Compute forces
	// For each particle q, parallelize
#pragma omp parallel for schedule(guided)
	for (unsigned int q = 0; q < N; q++) {
	    // for each particle k > q
	    double x_diff = 0.0;
	    double y_diff = 0.0;
	    double dist = 0.0;
	    double dist_cubed = 0.0;
	    double force_qk_X = 0.0;
	    double force_qk_Y = 0.0;
	    for (unsigned int k = q+1; k < N; k++) {
	        x_diff = pos[q][X] - pos[k][X]; 
	        y_diff = pos[q][Y] - pos[k][Y]; 
	        dist = sqrt(x_diff*x_diff + y_diff*y_diff); 
		dist_cubed = dist*dist*dist;
		force_qk_X = G*mass[q]*mass[k]/dist_cubed * x_diff;
	        force_qk_Y = G*mass[q]*mass[k]/dist_cubed * y_diff;
		forces[q][X] += force_qk_X;
		forces[q][Y] += force_qk_Y;
		// Protect from data race by using atomic directive
		// Ensures that the memory location is updated atomically
		#pragma omp atomic
		forces[k][X] -= force_qk_X;
		#pragma omp atomic
		forces[k][Y] -= force_qk_Y;
	    }
	}
	// Update velocities
	#pragma omp parallel for schedule(guided)
	for (unsigned int q = 0; q < N; q++) {
	    pos[q][X] += DT*vel[q][X];
	    pos[q][Y] += DT*vel[q][Y];
	    vel[q][X] += DT/mass[q]*forces[q][X];
	    vel[q][Y] += DT/mass[q]*forces[q][Y];	   	    
	}
    }
    // Print positions
    double elapsed_time = omp_get_wtime() - start_time;
#if PRINT
    for (unsigned int q = 0; q < N; q++)
	q == N - 1 ? printf("(%f, %f)\n", pos[q][X], pos[q][Y]) : printf("(%f, %f)", pos[q][X], pos[q][Y]);
    #endif
    free(forces);
    return elapsed_time;
}

void init_data(vect_t *pos, vect_t *old_pos, vect_t *vel, double *mass) {
    for (int i = 0; i < N; i++) {
	pos[i][X] = (rand() / (double)(RAND_MAX)) * 2 - 1;
	pos[i][Y] = (rand() / (double)(RAND_MAX)) * 2 - 1;

	old_pos[i][X] = pos[i][X];
	old_pos[i][Y] = pos[i][Y];

	vel[i][X] = (rand() / (double)(RAND_MAX)) * 2 - 1;
	vel[i][Y] = (rand() / (double)(RAND_MAX)) * 2 - 1;

	mass[i] = fabs((rand() / (double)(RAND_MAX)) * 2 - 1);
    }
}

int main(void) {
    // Init Particles
    vect_t *pos = malloc(N*sizeof(vect_t));
    vect_t *old_pos = malloc(N*sizeof(vect_t));
    vect_t *vel = malloc(N*sizeof(vect_t));
    double *mass = malloc(N*sizeof(double));
    

    
    // Simple algorithm,
    init_data(pos, old_pos, vel, mass);
    // Reduced algorithms
    vect_t *pos_red = malloc(N*sizeof(vect_t));;
    vect_t *old_pos_red = malloc(N*sizeof(vect_t));
    vect_t *vel_red = malloc(N*sizeof(vect_t));
    double *mass_red = malloc(N*sizeof(double));
    memcpy(pos_red, pos, N*sizeof(vect_t));
    memcpy(old_pos_red, old_pos, N*sizeof(vect_t));
    memcpy(vel_red, vel, N*sizeof(vect_t));
    memcpy(mass_red, mass, N*sizeof(double));
    
    double t_simple = simple_algorithm(pos, old_pos, vel, mass);
    printf("Time SIMPLE = %f\n", t_simple);

    double t_reduced = reduced_algorithm(pos_red, old_pos_red, vel_red, mass_red);
    printf("Time reduced = %f\n", t_reduced);
    
    // Cleanup
    free(pos); free(pos_red);
    free(old_pos); free(old_pos_red);
    free(vel); free(vel_red);
    free(mass); free(mass_red);

    return 0;
}
