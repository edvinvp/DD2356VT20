# DD2356 Project: Distributed Matrix-Matrix Multiply with Fox Algorithm

### Compile:
Compile on Beskow using the GNU environment with:
	cc -fopenmp -O2 fox.c -o fox
### Run:
To run the code interactively on Beskow first allocate *n* nodes for one hour:
       	salloc --nodes=*n* -t 01:00:00 -A edu20.DD2356 -C Haswell
Then run using *p* nodes with srun:
        srun -n *p* ./fox