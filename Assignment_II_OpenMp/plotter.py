# Author: Edvin von Platen
# Plotting and data analysis for assignment 2

import statistics as stat
from matplotlib import pyplot as plt

def exercise_1():
    threads = [1,2,4,8,12,16,20,24,28,32]

    g_1 = [11152.8, 10374.1, 11401.8, 10377.4, 11600.1]
    g_2 = [19838.8, 20192.2, 20045.1, 19962.8, 20082.9]
    g_4 = [35242.6, 34572.6, 35282.5, 35413.6, 35201.9]
    g_8 = [44457.7, 44296.3, 44906.9, 47382.1, 44284.6]
    g_12 = [45833.1, 45581.0, 45636.8, 45636.8, 45402.1]
    g_16 = [45739.4, 45965.0, 45895.8, 46084.9, 46056.5]
    g_20 = [45751.9, 45157.7, 45288.7, 45924.1, 44855.9]
    g_24 = [44718.4, 45543.9, 45325.5, 44867.9, 45689.6]
    g_28 = [45273.5, 45911.5, 45805.0, 46743.0, 45584.1]
    g_32 = [45057.7, 45805.0, 45325.5, 45780.0, 45596.5]

    guided_runs = [g_1, g_2, g_4, g_8, g_12, g_16, g_20, g_24, g_28, g_32]
    g_means = [stat.mean(x) for x in guided_runs]
    g_stdevs = [stat.stdev(x) for x in guided_runs]
    print(g_means)

    # STATIC - no chunk size specified (openmp decides)

    s_1 = [12049.1, 11766.2, 11153.0, 11753.5, 11146.0]
    s_2 = [23605.8, 22374.8, 21941.8, 22209.7, 22023.8]
    s_4 = [35266.6, 35413.6, 34934.3, 35383.8, 35352.1]
    s_8 = [40201.8, 40160.9, 40158.5, 40122.5, 40160.9]
    s_12 = [40559.0, 40640.0, 40691.8, 40743.6, 40785.7]
    s_16 = [40763.4, 40827.9, 37788.7, 40795.7, 40783.3]
    s_20 = [40536.9, 40559.0, 41971.9, 40514.9, 40517.3]
    s_24 = [40723.9, 40785.7, 42463.2, 40890.1, 44346.0]
    s_28 = [40701.6, 43679.3, 43656.6, 43537.6, 43645.2]
    s_32 = [43254.2, 49954.5, 47876.8, 46135.6, 44605.4]

    static_runs = [s_1, s_2, s_4, s_8, s_12, s_16, s_20, s_24, s_28, s_32]
    s_means = [stat.mean(x) for x in static_runs]
    s_stdevs = [stat.stdev(x) for x in static_runs]
    # DYNAMIC - no chunk size spcecified (defaults to size 1 chunks)
    d_1 = [1468.7, 1468.3, 1147.9, 1469.5, 1148.2]
    d_2 = [171.8, 180.9, 171.1, 143.3, 138.0]
    d_4 = [284.0, 220.5, 214.1, 260.9, 278.0]
    d_8 = [291.2, 314.3, 311.7, 242.0, 280.4]
    d_12 = [260.2, 255.2, 266.8, 216.2, 261.3]
    d_16 = [249.0, 282.7, 277.6, 247.7, 280.5]
    d_20 = [218.0, 387.9, 355.4, 339.2, 270.8]
    d_24 = [452.2, 342.1, 213.1, 338.2, 294.8]
    d_28 = [303.3, 354.9, 408.2, 538.4, 450.1]
    d_32 = [288.3, 552.1, 304.1, 272.6, 531.8]
    dynamic_runs = [d_1, d_2, d_4, d_8, d_12, d_16, d_20, d_24, d_28, d_32]
    d_means = [stat.mean(x) for x in dynamic_runs]
    d_stdevs = [stat.stdev(x) for x in dynamic_runs]


    plt.plot(threads,g_means, linestyle='--', marker='o', label='Guided')
    plt.plot(threads,d_means, linestyle='--', marker='o', label='Dynamic')
    plt.plot(threads,s_means, linestyle='--', marker='o', label='Static')
    plt.legend()
    plt.xlabel('Threads')
    plt.ylabel('Stream Copy MB/s')
    plt.title('OpenMP Scheduling Stream Copy Benchmark')
    plt.show()

    def make_table(means, stdevs, threads):
        n = len(means)
        table_str = "\\begin{tabular} {"
        cols = "c "*(n+1)
        table_str += cols
        table_str += '}\n'
        # row 1: threads
        table_str += "THREADS: & " +  " & ".join(str(e) for e in threads) + ' \\\\ \n'
        table_str += "MEAN MB/s: & " +  " & ".join(str(round(e,2)) for e in means) + ' \\\\ \n'
        table_str += "STDEV MB/s: & " +  " & ".join(str(round(e,2)) for e in stdevs)
        return table_str + "\n \\end{tabular}\n"
    
    print(make_table(g_means[0:6], g_stdevs[0:6], threads[0:6]))
    print(make_table(g_means[6:], g_stdevs[6:], threads[6:]))
    print()
    print(make_table(d_means[0:6], d_stdevs[0:6], threads[0:6]))
    print(make_table(d_means[6:], d_stdevs[6:], threads[6:]))
    print()
    print(make_table(s_means[0:6], s_stdevs[0:6], threads[0:6]))
    print(make_table(s_means[6:], s_stdevs[6:], threads[6:]))



def exercise_3_O2():
    threads = [1,2,4,8,12,16,20,24,28,32]
    
    Q1 = [0.001053, 0.001050, 0.001023, 0.001023, 0.001023]
    Q1_mean = stat.mean(Q1)
    Q1_stdev = stat.stdev(Q1)
    print("Q1 mean + STDEV")
    print(round(Q1_mean, 6))
    print(round(Q1_stdev, 6))
    
    Q2 = [0.001129, 0.000124, 0.000117, 0.000122, 0.000115]
    Q2_mean = stat.mean(Q2)
    Q2_stdev = stat.stdev(Q2)
    print("Q2 mean + STDEV")
    print(round(Q2_mean, 6))
    print(round(Q2_stdev, 6))
    
    Q3_1 = [0.015426, 0.015461, 0.015432, 0.015431, 0.015430]
    Q3_2 = [0.031988, 0.031144, 0.031614, 0.031311, 0.031220]
    Q3_4 = [0.049798, 0.049253, 0.048181, 0.039267, 0.040812]
    Q3_8 = [0.074931, 0.074083, 0.077666, 0.067043, 0.080285]
    Q3_12 = [0.068380, 0.064826, 0.074451, 0.073036, 0.069136]
    Q3_16 = [0.083668, 0.078051, 0.072178, 0.074251, 0.073893]
    Q3_20 = [0.092010, 0.068846, 0.074729, 0.067525, 0.077678]
    Q3_24 = [0.097372, 0.097769, 0.098732, 0.102362, 0.101020]
    Q3_28 = [0.114609, 0.108294, 0.120094, 0.125797, 0.126202]
    Q3_32 = [0.147645, 0.159570, 0.159482, 0.161813, 0.152697]
    
    Q3_runs = [Q3_1, Q3_2, Q3_4, Q3_8, Q3_12, Q3_16, Q3_20, Q3_24, Q3_28, Q3_32]
    Q3_means = [stat.mean(x) for x in Q3_runs]
    Q3_stdevs = [stat.stdev(x) for x in Q3_runs]
    Q4_1 = [0.000595, 0.000593, 0.000593, 0.000593, 0.000561]
    Q4_2 = [0.000327, 0.000535, 0.000535, 0.000522, 0.000672]
    Q4_4 = [0.000472, 0.000328, 0.000295, 0.000333, 0.000519]
    Q4_8 = [0.000190, 0.000248, 0.000317, 0.000255, 0.000268]
    Q4_12 = [0.000219, 0.000216, 0.000223, 0.000196, 0.000244]
    Q4_16 = [0.000187, 0.000222, 0.000211, 0.000207, 0.000206]
    Q4_20 = [0.000151, 0.000154, 0.000156, 0.000138, 0.000159]
    Q4_24 = [0.000179, 0.000164, 0.000175, 0.000179, 0.000178]
    Q4_28 = [0.000172, 0.000211, 0.000178, 0.000165, 0.000221]
    Q4_32 = [0.000160, 0.000182, 0.000194, 0.000178, 0.000175]
    
    Q4_runs = [Q4_1, Q4_2, Q4_4, Q4_8, Q4_12, Q4_16, Q4_20, Q4_24, Q4_28, Q4_32]
    Q4_means = [stat.mean(x) for x in Q4_runs]
    Q4_stdevs = [stat.stdev(x) for x in Q4_runs]

    Q5_1 = [0.000630, 0.000594, 0.000594, 0.000594, 0.000562]
    Q5_2 = [0.000328, 0.000554, 0.000674, 0.000529, 0.000586]
    Q5_4 = [0.000293, 0.000327, 0.000534, 0.000427, 0.000547]
    Q5_8 = [0.000227, 0.000212, 0.000310, 0.000260, 0.000274]
    Q5_12 = [0.000243, 0.000238, 0.000203, 0.000247, 0.000240]
    Q5_16 = [0.000200, 0.000199, 0.000162, 0.000199, 0.000160]
    Q5_20 = [0.000145, 0.000154, 0.000145, 0.000157, 0.000189]
    Q5_24 = [0.000171, 0.000173, 0.000195, 0.000174, 0.000172]
    Q5_28 = [0.000163, 0.000204, 0.000166, 0.000169, 0.000209]
    Q5_32 = [0.000168, 0.000170, 0.000177, 0.000176, 0.000174]
    Q5_runs = [Q5_1, Q5_2, Q5_4, Q5_8, Q5_12, Q5_16, Q5_20, Q5_24, Q5_28, Q5_32]
    Q5_means = [stat.mean(x) for x in Q5_runs]
    Q5_stdevs = [stat.stdev(x) for x in Q5_runs]
    
    plt.plot(threads,Q3_means, linestyle='--', marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Time Seconds (s)')
    plt.title('Maxloc with Critical Section using -O2 flag')
    plt.show()
    
    plt.plot(threads,Q4_means, linestyle='--', marker='o', label='No Critical Section')
    plt.plot(threads,Q5_means, linestyle='--', marker='o', label='Padded Struct')
    plt.legend()
    plt.xlabel('Threads')
    plt.ylabel('Time Seconds (s)')
    plt.title('Maxloc comparision using -O2 flag')
    plt.show()
    
def exercise_3_no_O2():
    threads = [1,2,4,8,12,16,20,24,28,32]
    
    Q1 = [0.002001, 0.001921, 0.001816, 0.001815, 0.001815]
    Q1_mean = stat.mean(Q1)
    Q1_stdev = stat.stdev(Q1)
    
    Q2 = [0.004710, 0.000299, 0.000198, 0.000223, 0.000191]
    Q2_mean = stat.mean(Q2)
    Q2_stdev = stat.stdev(Q2)
    
    Q3_1 = [0.014549, 0.014460, 0.014463, 0.014461, 0.014463]
    Q3_2 = [0.039265, 0.037639, 0.039128, 0.038371, 0.038625]
    Q3_4 = [0.074220, 0.074271, 0.073367, 0.074335, 0.058198]
    Q3_8 = [0.104536, 0.100370, 0.100645, 0.103836, 0.101721]
    Q3_12 = [0.117362, 0.112064, 0.117478, 0.112686, 0.118717]
    Q3_16 = [0.137478, 0.148016, 0.143499, 0.150515, 0.150003]
    Q3_20 = [0.160740, 0.156994, 0.171839, 0.172941, 0.171355]
    Q3_24 = [0.181282, 0.192880, 0.185340, 0.185727, 0.188794]
    Q3_28 = [0.201532, 0.210470, 0.212016, 0.209253, 0.215629]
    Q3_32 = [0.223233, 0.223653, 0.222246, 0.221695, 0.222240]

    Q3_runs = [Q3_1, Q3_2, Q3_4, Q3_8, Q3_12, Q3_16, Q3_20, Q3_24, Q3_28, Q3_32]
    Q3_means = [stat.mean(x) for x in Q3_runs]
    Q3_stdevs = [stat.stdev(x) for x in Q3_runs]

    Q4_1 = [0.002238, 0.002234, 0.002238, 0.002238, 0.002234]
    Q4_2 = [0.001162, 0.001378, 0.001537, 0.001524, 0.001378]
    Q4_4 = [0.000676, 0.000633, 0.000822, 0.000671, 0.000851]
    Q4_8 = [0.000425, 0.000556, 0.000402, 0.000406, 0.000548]
    Q4_12 = [0.000340, 0.000337, 0.000346, 0.000352, 0.000341]
    Q4_16 = [0.000293, 0.000315, 0.000311, 0.000309, 0.000544]
    Q4_20 = [0.000311, 0.000296, 0.000349, 0.000430, 0.000346]
    Q4_24 = [0.000393, 0.000306, 0.000349, 0.000362, 0.000468]
    Q4_28 = [0.000331, 0.000309, 0.000470, 0.000392, 0.000406]
    Q4_32 = [0.000374, 0.000322, 0.000315, 0.000409, 0.000395]
    Q4_runs = [Q4_1, Q4_2, Q4_4, Q4_8, Q4_12, Q4_16, Q4_20, Q4_24, Q4_28, Q4_32]
    Q4_means = [stat.mean(x) for x in Q4_runs]
    Q4_stdevs = [stat.stdev(x) for x in Q4_runs]

    Q5_1 = [0.002291, 0.002291, 0.002294, 0.002293, 0.002292]
    Q5_2 = [0.001296, 0.001501, 0.001293, 0.001316, 0.001526]
    Q5_4 = [0.000995, 0.000692, 0.000760, 0.000724, 0.000832]
    Q5_8 = [0.000595, 0.000431, 0.000495, 0.000543, 0.000574]
    Q5_12 = [0.000417, 0.000457, 0.000428, 0.000400, 0.000456]
    Q5_16 = [0.000347, 0.000453, 0.000295, 0.000294, 0.000535]
    Q5_20 = [0.000265, 0.000257, 0.000277, 0.000391, 0.000423]
    Q5_24 = [0.000269, 0.000389, 0.000271, 0.000383, 0.000356]
    Q5_28 = [0.000367, 0.000322, 0.000362, 0.000323, 0.000266]
    Q5_32 = [0.000341, 0.000280, 0.000286, 0.000246, 0.000335]
    Q5_runs = [Q5_1, Q5_2, Q5_4, Q5_8, Q5_12, Q5_16, Q5_20, Q5_24, Q5_28, Q5_32]
    Q5_means = [stat.mean(x) for x in Q5_runs]
    Q5_stdevs = [stat.stdev(x) for x in Q5_runs]
    
    plt.plot(threads,Q3_means, linestyle='--', marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Time Seconds (s)')
    plt.title('Maxloc with Critical Section without -O2 flag')
    plt.show()
    
    plt.plot(threads,Q4_means, linestyle='--', marker='o', label='No Critical Section')
    plt.plot(threads,Q5_means, linestyle='--', marker='o', label='Padded Struct')
    plt.legend()
    plt.xlabel('Threads')
    plt.ylabel('Time Seconds (s)')
    plt.title('Maxloc comparision without -O2 flag')
    plt.show()

def exercise_4():
    threads = [1,2,4,8,12,16,20,24,28,32]
    t_1 =  [5.540693, 6.052120, 6.147383, 6.075304, 6.136308]
    t_2 =  [2.770642, 2.769640, 3.381095, 3.954810, 3.402137]
    t_4 =  [1.783430, 1.785847, 1.472497, 1.462698, 1.451417]
    t_8 =  [0.902501, 0.840305, 0.906079, 0.836278, 0.855967]
    t_12 = [0.603976, 0.605137, 0.605205, 0.604802, 0.604552]
    t_16 = [0.466414, 0.466946, 0,.48673, 0.467117, 0.471353]
    t_20 = [0.476247, 0.436715, 0.461263, 0.455988, 0.421684]
    t_24 = [0.383834, 0.407314, 0.409528, 0.410738, 0.404113]
    t_28 = [0.373150, 0.359981, 0.372484, 0.376732, 0.363960]
    t_32 = [0.342133, 0.323164, 0.339989, 0.335929, 0.329621]
    runs = [t_1, t_2, t_4, t_8, t_12, t_16, t_20, t_24, t_28, t_32]
    means = [stat.mean(x) for x in runs]
    
    plt.plot(threads,means, linestyle='--', marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Time Seconds (s)')
    plt.title('Question 5: DFTW')
    plt.show()

    print(stat.mean(t_32))
    print(stat.stdev(t_32))

def exercise_5():
    threads = [1,2,4,8,12,16,20,24,28,32]
    s_500_simple   = [0.197379, 0.250888, 0.243394, 0.238275, 0.243268]
    s_500_s_mean = stat.mean(s_500_simple)
    s_500_s_stdev = stat.stdev(s_500_simple)
    s_500_reduced  = [0.098732, 0.125596, 0.125660, 0.125603, 0.125624]
    s_500_r_mean = stat.mean(s_500_reduced)
    s_500_r_stdev = stat.stdev(s_500_reduced)
    s_1000_simple  = [0.996827, 1.001411, 0.996763, 0.787939, 0.993710]
    s_1000_s_mean = stat.mean(s_1000_simple)
    s_1000_s_stdev = stat.stdev(s_1000_simple)
    s_1000_reduced = [0.501782, 0.504179, 0.501822, 0.391199, 0.501784]
    s_1000_r_mean = stat.mean(s_1000_reduced)
    s_1000_r_stdev = stat.stdev(s_1000_reduced)
    s_2000_simple  = [3.735428, 3.145232, 3.179680, 3.130821, 3.745162]
    s_2000_s_mean = stat.mean(s_2000_simple)
    s_2000_s_stdev = stat.stdev(s_2000_simple)
    s_2000_reduced = [1.561793, 1.564298, 1.561458, 1.561458, 1.561774]
    s_2000_r_mean = stat.mean(s_2000_reduced)
    s_2000_r_stdev = stat.stdev(s_2000_reduced)
    print("500 simple / reduced")
    print("mean simple : " + str(s_500_s_mean))
    print("stdev simple : " + str(s_500_s_stdev))
    print("mean reduced : " + str(s_500_r_mean))
    print("stdev reduced : " + str(s_500_r_stdev))
    print("1000 simple / reduced")
    print("mean simple : " + str(s_1000_s_mean))
    print("stdev simple : " + str(s_1000_s_stdev))
    print("mean reduced : " + str(s_1000_r_mean))
    print("stdev reduced : " + str(s_1000_r_stdev))
    print("2000 simple / reduced")
    print("mean simple : " + str(s_2000_s_mean))
    print("stdev simple : " + str(s_2000_s_stdev))
    print("mean reduced : " + str(s_2000_r_mean))
    print("stdev reduced : " + str(s_2000_r_stdev))

    # n = 1000, dt = 0.05
    # Fastest from 3 runs
    ps_1 = 1.632435
    pr_1 = 0.578431
    ps_2 = 1.081370
    pr_2 = 1.125693
    ps_4 = 0.471715
    pr_4 = 1.121209
    ps_8 = 0.290731
    pr_8 = 1.250799
    ps_12 = 0.223403
    pr_12 = 1.237125
    ps_16 = 0.187745
    pr_16 = 1.275715
    ps_20 = 0.142408
    pr_20 = 1.411904
    ps_24 = 0.134068
    pr_24 = 1.457811
    ps_28 = 0.130929
    pr_28 = 1.621283
    ps_32 = 0.137725
    pr_32 = 1.746516
    ps_data = [ps_1, ps_2, ps_4, ps_8, ps_12, ps_16, ps_20, ps_24, ps_28, ps_32]
    pr_data = [pr_1, pr_2, pr_4, pr_8, pr_12, pr_16, pr_20, pr_24, pr_28, pr_32]
    plt.plot(threads,ps_data, linestyle='--', marker='o', label='Simple Algortihm')
    plt.plot(threads,pr_data, linestyle='--', marker='o', label='Reduced Algorithm')
    plt.legend()
    plt.xlabel('Threads')
    plt.ylabel('Time Seconds (s)')
    plt.title('Parallelized N-Body Measurement.')
    plt.show()
    # all cyclic/dynamic chunk size = 50 N = 1000, dt = 0.05, 32 threads
    ps_cyclic = [0.162676, 0.164876, 0.161363, 0.167700, 0.161664]
    pr_cyclic = [1.340191, 1.754834, 1.817688, 1.743119, 1.673199]
    ps_cyclic_mean = stat.mean(ps_cyclic)
    ps_cyclic_stdev = stat.stdev(ps_cyclic)
    pr_cyclic_mean = stat.mean(pr_cyclic)
    pr_cyclic_stdev = stat.stdev(pr_cyclic)

    # all static/blocked scheduling
    ps_static = [0.120498, 0.119189, 0.122209, 0.113881, 0.116570]
    pr_static = [1.514852, 1.349016, 1.479090, 1.289755, 1.255786]
    ps_static_mean = stat.mean(ps_static)
    ps_static_stdev = stat.stdev(ps_static)
    pr_static_mean = stat.mean(pr_static)
    pr_static_stdev = stat.stdev(pr_static)

    print("ps_cyclic_mean = " + str(ps_cyclic_mean))
    print("ps_cyclic_stdev = " + str(ps_cyclic_stdev))
    print("pr_cyclic_mean = " + str(pr_cyclic_mean))
    print("pr_cyclic_stdev = " + str(pr_cyclic_stdev))
    print("ps_static_mean = " + str(ps_static_mean))
    print("ps_static_stdev = " + str(ps_static_stdev))
    print("pr_static_mean = " + str(pr_static_mean))
    print("pr_static_stdev = " + str(pr_static_stdev))


    
#exercise_3_O2()
#exercise_3_no_O2()
#exercise_4()
#exercise_5()
