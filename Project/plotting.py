from matplotlib import pyplot as plt

def matrix_test():
    ### Plot data from matrix multiply benchmarks
    n = [32, 64, 128, 256, 512, 1024, 2048, 4096]

   # naive =   [0.000029, 0.000277, 0.002561, 0.025223, 0.286924, 2.767771]
   # reorder = [0.000019, 0.000162, 0.001231, 0.009348, 0.068575, 0.540787]
    unroll =  [0.000016, 0.000123, 0.001072, 0.008469, 0.059290, 0.468197, 3.742584, 36.976377]
    omp_4 = [0.000020, 0.000058, 0.000340, 0.002369, 0.017562, 0.133803,  0.986958, 10.511707]
    strip_4 = [ 0.000013, 0.000040, 0.000279, 0.002036, 0.018696,  0.155797, 1.238403, 11.375591]
    omp_8 = [0.000056, 0.000050, 0.000186, 0.001332, 0.010855, 0.100954, 0.765742, 5.569788]
    strip_8 = [0.000031,  0.000048, 0.000166,  0.001219, 0.011541, 0.106149, 0.878457, 5.680306]

   # plt.plot(n, naive, linestyle='--', marker='o', label='1: Naive');
   # plt.plot(n, reorder, linestyle='--', marker='o', label='2: Reorder');
   # plt.plot(n, unroll, linestyle='--', marker='o', label='3: Unroll');
    plt.plot(n, omp_4, linestyle='--', marker='o', label='4: OpenMP 4 threads');
    plt.plot(n, omp_8, linestyle='--', marker='o', label='4: OpenMP 8 threads');
    plt.plot(n, strip_4, linestyle='--', marker='o', label='5: Tiling 4 threads');
    plt.plot(n, strip_8, linestyle='--', marker='o', label='5: Tiling 8 threads');

    plt.legend()
    plt.xlabel('n: Matrix size n*n')
    plt.ylabel("Runtime sec")
    plt.show()

def fox_plot():
    # Ran on even powers of 2
    processes_10 = [4, 16, 64]
    processes_12 = [4, 16, 64, 256]
    processes_14 = [4, 16, 64, 256]

    # 2^n x 2^n matrix size
    # Block size is then 2^n/sqrt(#processes)
    n_10 = [0.030201, 0.018502, 0.019630]
    n_12 = [1.538337, 0.351636, 0.166091, 0.168381]
    n_14 = [89.883720, 22.586349, 5.715725, 5.675043] 

    plt.plot(processes_10, n_10, linestyle='--', marker='o')
    plt.xlabel("Number of processes")
    plt.ylabel("Runtime sec")
    plt.show()
    plt.plot(processes_12, n_12, linestyle='--', marker='o')
    plt.xlabel("Number of processes")
    plt.ylabel("Runtime sec")
    plt.show()
    plt.plot(processes_14, n_14, linestyle='--', marker='o')
    plt.xlabel("Number of processes")
    plt.ylabel("Runtime sec")
    plt.show()
    

#matrix_test()
fox_plot()
